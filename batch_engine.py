"""Micro-batching wrapper for the in-process vLLM engine.

Lets k voting trajectories (one thread each) share one engine efficiently:
each thread's per-step ``.chat`` call blocks until every *live* trajectory
has a request queued, then the last arriver submits them all as ONE batched
``engine.chat`` call and distributes the outputs. vLLM decodes the batch in
parallel, so k votes cost roughly one trajectory's wall time, not k.

A trajectory that finishes early calls :meth:`unregister`, lowering the bar
so the remaining trajectories never wait on it. With one registered worker
every call flushes immediately, so the wrapper is also safe at votes=1.

Only the flushing thread ever touches the underlying engine, so the offline
vLLM engine never sees concurrent calls.
"""

import threading
from typing import Dict, List


class BatchedEngine:
    """Drop-in for the ``.chat(messages=[conv], sampling_params=sp)`` calls
    :class:`gvd.vllm_llm.VLLMToolClient` makes (single conversation per call).
    Requests may carry different SamplingParams; they are passed per-prompt."""

    def __init__(self, engine):
        self._engine = engine
        self._cv = threading.Condition()
        self._live = 0
        self._pending: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Worker lifecycle — register BEFORE the worker threads start, so an
    # early-arriving request can't see live=1 and flush a lonely batch.
    # ------------------------------------------------------------------ #

    def register(self, n: int = 1):
        with self._cv:
            self._live += n

    def unregister(self):
        with self._cv:
            self._live -= 1
            batch = self._take_batch_locked()
        if batch:
            self._run(batch)

    # ------------------------------------------------------------------ #

    def chat(self, messages, sampling_params, **kwargs):
        slot = {"event": threading.Event(), "out": None, "err": None}
        with self._cv:
            self._pending.append({"conv": messages[0], "sp": sampling_params,
                                  "kw": kwargs, "slot": slot})
            batch = self._take_batch_locked()
        if batch:
            self._run(batch)
        slot["event"].wait()
        if slot["err"] is not None:
            raise slot["err"]
        return [slot["out"]]

    def _take_batch_locked(self):
        # Each live worker has at most one request in flight, so pending can
        # only reach live when every remaining trajectory is waiting here.
        if self._pending and len(self._pending) >= self._live:
            batch, self._pending = self._pending, []
            return batch
        return None

    def _run(self, batch: List[Dict]):
        kw = dict(batch[0]["kw"])
        kw.pop("use_tqdm", None)
        try:
            outs = self._engine.chat(
                messages=[r["conv"] for r in batch],
                sampling_params=[r["sp"] for r in batch],
                use_tqdm=False, **kw)
            for r, o in zip(batch, outs):
                r["slot"]["out"] = o
                r["slot"]["event"].set()
        except Exception as exc:
            for r in batch:
                r["slot"]["err"] = exc
                r["slot"]["event"].set()
