"""Open-vocabulary object detector for precise counting & spatial reasoning.

The vision-language model is good at description but unreliable at *counting* —
it eyeballs "about four" where the truth is six. A dedicated detector grounds
the count in actual bounding boxes: prompt it with a phrase ("person carrying a
sedan chair"), get one box per instance, count the boxes.

Wraps GroundingDINO (open-vocabulary, phrase-grounded) via 🤗 transformers. The
model is loaded lazily on first use and cached, so importing this module is
free and sessions that never count never pay for it.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class GroundingDinoDetector:
    """Lazy open-vocabulary detector. Counts instances of a text phrase."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: Optional[str] = None):
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = (AutoModelForZeroShotObjectDetection
                       .from_pretrained(self.model_id).to(self._device).eval())

    def detect(self, image, phrases: List[str],
               box_threshold: float = 0.30,
               text_threshold: float = 0.25) -> List[Dict]:
        """Return [{'phrase','score','box'}] for one PIL image.

        GroundingDINO wants a lowercase, '.'-terminated text query; multiple
        phrases are joined so a single forward pass grounds them all.
        """
        import torch
        self._ensure_loaded()
        text = ". ".join(p.strip().lower().rstrip(".") for p in phrases if p.strip()) + "."
        inputs = self._processor(images=image, text=text,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # transformers renamed the kwarg (box_threshold→threshold) across versions.
        post = self._processor.post_process_grounded_object_detection
        target_sizes = [image.size[::-1]]  # (h, w)
        try:
            res = post(outputs, inputs.input_ids, threshold=box_threshold,
                       text_threshold=text_threshold, target_sizes=target_sizes)[0]
        except TypeError:
            res = post(outputs, inputs.input_ids, box_threshold=box_threshold,
                       text_threshold=text_threshold, target_sizes=target_sizes)[0]
        labels = res.get("text_labels", res.get("labels", []))
        dets = []
        for box, score, lab in zip(res["boxes"], res["scores"], labels):
            dets.append({
                "phrase": lab if isinstance(lab, str) else str(lab),
                "score": float(score),
                "box": [round(float(c), 1) for c in box.tolist()],
            })
        return dets
