import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from vllm import LLM, SamplingParams

try:
    from .load_data import safe_open_image as _safe_open_image
except Exception:
    try:
        from load_data import safe_open_image as _safe_open_image
    except Exception:
        _safe_open_image = None


INSTRUCTION = '''You are an image description expert. You are given an original image and manipulation text. Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.

## Guidelines on generating the Original Image Description
- Ensure the original image description is thorough, capturing all visible objects, attributes, and elements.
- The original image description should be as accurate as possible, reflecting the content of the image.

## Guidelines on generating the Thoughts
- In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target image description.
- Provide insight into how you interpreted the manipulation intent in detail in the manipulation text.
- Discuss how the manipulation intent influenced which elements of the original image you focused on.

## Guidelines on generating the Reflections
- In your Reflections, summarize how the manipulation intent influenced your approach to transforming the original image description.
- Explain how the changes made reflect the specific semantic, Highlight key decisions that were made to preserve the coherence and context of the original image while meeting the manipulation intent.
- Reflect on the impact these changes have on the overall appearance or narrative of the image.
- Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies applied in crafting the target description, offering a logical connection between the original and final content.

## Guidelines on generating Target Image Description
- The target image description you generate should be complete and can cover various semantic aspects.
- The target image description only contains the target image content and needs to be as simple as possible. Minimize aesthetic descriptions as much as possible.

## On the input format <Input>
- Input consist of two parts: The original image and the manipulation text.
{
  "Original Image": <image_url>,
  "Manipulation Text": <manipulation_text>
}

## Guidelines on determining the response <Response>
- Responses include the Original Image Context, Target Image Description, and Thoughts.
{
  "Original Image Description": <original_image_description>,
  "Thoughts": <thoughts>,
  "Reflections": <reflections>,
  "Target Image Description": <target_image_description>
}'''


class VisualICL:
    SUPPORTED_METHODS = {"none", "random", "rices", "muier", "mmices", "circles"}

    def __init__(
        self,
        vlm: Optional[LLM] = None,
        method: str = "none",
        clip_model_name: str = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        sampling_params: Optional[SamplingParams] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_tqdm: bool = False,
        default_k: int = 16,
        num_attributes: int = 1,
        attribute_k: int = 16,
        seed: int = 42,
        llm: Optional[LLM] = None,
    ) -> None:
        if vlm is None:
            vlm = llm
        if vlm is None:
            raise ValueError("`vlm` must be provided.")
        self.vlm = vlm
        self.llm = self.vlm
        self.method = (method or "none").lower()
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown retrieval method: {self.method}")
        self.clip_model_name = clip_model_name
        self.sampling_params = sampling_params or SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.use_tqdm = use_tqdm
        self.default_k = default_k
        self.num_attributes = num_attributes
        self.attribute_k = attribute_k
        self._clip_model: Optional[CLIPModel] = None
        self._clip_processor: Optional[CLIPProcessor] = None
        random.seed(seed)
        np.random.seed(seed)

    @classmethod
    def from_model(
        cls,
        model: str,
        method: str = "none",
        clip_model_name: str = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        default_k: int = 16,
        num_attributes: int = 1,
        attribute_k: int = 16,
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_tqdm: bool = False,
        max_model_len: int = 40960,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        limit_mm_per_prompt: int = 36,
        mm_processor_cache_gb: float = 0,
        trust_remote_code: bool = True,
        seed: int = 42,
    ) -> "VisualICL":
        vlm = LLM(
            model=model,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": limit_mm_per_prompt, "video": 0},
            mm_processor_cache_gb=mm_processor_cache_gb,
            trust_remote_code=trust_remote_code,
        )
        return cls(
            vlm=vlm,
            method=method,
            clip_model_name=clip_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=use_tqdm,
            default_k=default_k,
            num_attributes=num_attributes,
            attribute_k=attribute_k,
            seed=seed,
        )

    def predict(
        self,
        question: str,
        query_image: Union[str, Image.Image],
        train_dataset: Any,
        k: Optional[int] = None,
        task: str = "vqa",
        options: Optional[Sequence[str]] = None,
        num_attributes: Optional[int] = None,
        attribute_k: Optional[int] = None,
        attributes: Optional[Sequence[str]] = None,
        retrieved_examples: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not question or not isinstance(question, str):
            raise ValueError("`question` must be a non-empty string.")
        if train_dataset is None or not hasattr(train_dataset, "__len__"):
            raise ValueError("`train_dataset` must be a non-empty dataset-like object.")
        if len(train_dataset) == 0:
            raise ValueError("`train_dataset` is empty.")

        method = self.method
        max_side = self._dataset_max_side(train_dataset)

        image = self._to_image(query_image, max_side=max_side)
        if image is None:
            raise ValueError("`query_image` is invalid or could not be loaded.")
        k = self._safe_k(k if k is not None else self.default_k, len(train_dataset))
        n_attr = max(1, int(num_attributes if num_attributes is not None else self.num_attributes))
        a_k = self._safe_k(attribute_k if attribute_k is not None else self.attribute_k, len(train_dataset))
        provided_attributes = self._normalize_attributes(attributes, n_attr)

        if retrieved_examples is None:
            retrieved_examples = self.retrieve_examples(
                image=image,
                question=question,
                train_dataset=train_dataset,
                method=method,
                k=k,
                num_attributes=n_attr,
                attribute_k=a_k,
                attributes=provided_attributes,
            )
        else:
            retrieved_examples = self._coerce_retrieved_examples(retrieved_examples=retrieved_examples, method=method)

        messages = [[{"role": "user", "content": self._build_prompt_content(
            question=question,
            query_image=image,
            retrieved_examples=retrieved_examples,
            method=method,
            task=task,
            options=options,
            k=k,
            num_attributes=n_attr,
            attribute_k=a_k,
            max_side=max_side,
        )}]]

        outputs = self.vlm.chat(messages, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm)
        answer = ""
        if outputs and getattr(outputs[0], "outputs", None):
            answer = outputs[0].outputs[0].text

        return {
            "answer": answer,
            "retrieved_examples": retrieved_examples,
            "method": method,
            "k": k,
            "num_attributes": n_attr,
            "attribute_k": a_k,
            "attributes": self._extract_used_attributes(method=method, retrieved_examples=retrieved_examples),
        }

    def retrieve_examples(
        self,
        image: Image.Image,
        question: str,
        train_dataset: Any,
        method: str,
        k: int,
        num_attributes: int,
        attribute_k: int,
        attributes: Optional[Sequence[str]] = None,
    ) -> Any:
        if method == "none":
            return self.none(train_dataset=train_dataset)
        if method == "random":
            return self.random(train_dataset=train_dataset, k=k)
        if method == "rices":
            return self.rices(image=image, train_dataset=train_dataset, k=k)
        if method == "muier":
            return self.muier(image=image, train_dataset=train_dataset, k=k)
        if method == "mmices":
            return self.mmices(image=image, question=question, train_dataset=train_dataset, k=k)
        if method == "circles":
            return self.circles(
                image=image,
                question=question,
                train_dataset=train_dataset,
                k=k,
                num_attributes=num_attributes,
                attribute_k=attribute_k,
                attributes=attributes,
            )
        raise ValueError(f"Unknown retrieval method: {method}")

    def none(self, train_dataset: Any) -> List[Dict[str, Any]]:
        return []

    def random(self, train_dataset: Any, k: int) -> List[Dict[str, Any]]:
        k = self._safe_k(k, len(train_dataset))
        if k == 0:
            return []
        return [train_dataset[i] for i in random.sample(range(len(train_dataset)), k)]

    def rices(self, image: Image.Image, train_dataset: Any, k: int) -> List[Dict[str, Any]]:
        self._ensure_clip()
        self._ensure_dataset_image_embeddings(train_dataset)
        image_features = self._get_image_features(image)
        dataset_embeddings = train_dataset.image_embeddings.to(image_features.device)
        similarities = (image_features @ dataset_embeddings.T).squeeze(0)
        top_indices = self._top_indices(similarities, k)
        return [train_dataset[i] for i in top_indices]

    def muier(self, image: Image.Image, train_dataset: Any, k: int) -> List[Dict[str, Any]]:
        self._ensure_clip()
        self._ensure_dataset_image_embeddings(train_dataset)
        self._ensure_dataset_text_embeddings(train_dataset)

        image_features = self._get_image_features(image)
        img_embeddings = train_dataset.image_embeddings.to(image_features.device)
        txt_embeddings = train_dataset.text_embeddings.to(image_features.device)

        sim = (image_features @ img_embeddings.T).squeeze(0) + (image_features @ txt_embeddings.T).squeeze(0)
        top_indices = self._top_indices(sim, k)
        return [train_dataset[i] for i in top_indices]

    def mmices(self, image: Image.Image, question: str, train_dataset: Any, k: int) -> List[Dict[str, Any]]:
        self._ensure_clip()
        self._ensure_dataset_image_embeddings(train_dataset)

        image_features = self._get_image_features(image)
        question_features = self._get_text_features(question)

        img_embeddings = train_dataset.image_embeddings.to(image_features.device)
        sim_img = (image_features @ img_embeddings.T).squeeze(0)

        candidate_pool_size = min(max(1024, k * 8), len(train_dataset))
        candidate_idx = self._top_indices(sim_img, candidate_pool_size)
        candidate_embeddings = img_embeddings[candidate_idx]

        rerank = (question_features @ candidate_embeddings.T).squeeze(0)
        local_top = self._top_indices(rerank, k)
        final_idx = [candidate_idx[i] for i in local_top]
        return [train_dataset[i] for i in final_idx]

    def circles(
        self,
        image: Image.Image,
        question: str,
        train_dataset: Any,
        k: int,
        num_attributes: int,
        attribute_k: int,
        attributes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        self._ensure_clip()
        self._ensure_dataset_image_embeddings(train_dataset)
        self._ensure_dataset_text_embeddings(train_dataset)

        attr_list = self._normalize_attributes(attributes, num_attributes)
        if not attr_list:
            attr_list = self._identify_attributes(image=image, question=question, num_attributes=num_attributes)
        original_retrievals = self.rices(image=image, train_dataset=train_dataset, k=k)

        composed_retrievals: List[Dict[str, Any]] = []
        for attr in attr_list:
            caption = self._generate_modified_caption(image=image, attribute=attr)
            if not caption:
                composed_retrievals.append(
                    {
                        "attribute": attr,
                        "modified_caption": "",
                        "retrieved_items": [],
                    }
                )
                continue

            caption_features = self._get_text_features(caption)
            task_features = self._get_text_features(question)

            img_embeddings = train_dataset.image_embeddings.to(caption_features.device)
            txt_embeddings = train_dataset.text_embeddings.to(caption_features.device)
            sim = (caption_features @ img_embeddings.T).squeeze(0) + (task_features @ txt_embeddings.T).squeeze(0)
            top_indices = self._top_indices(sim, attribute_k)
            items = [train_dataset[i] for i in top_indices]
            composed_retrievals.append(
                {
                    "attribute": attr,
                    "modified_caption": caption,
                    "retrieved_items": items,
                }
            )

        return {
            "original_retrievals": original_retrievals,
            "composed_retrievals": composed_retrievals,
        }

    def _build_prompt_content(
        self,
        question: str,
        query_image: Image.Image,
        retrieved_examples: Any,
        method: str,
        task: str,
        options: Optional[Sequence[str]],
        k: int,
        num_attributes: int,
        attribute_k: int,
        max_side: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        task_name = "Visual Question Answering" if task == "vqa" else "Image Classification"
        prefix = f"Your task is to perform {task_name}."
        if task == "cls" and options:
            prefix += f" You need to choose one of the following options: {list(options)}"

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prefix},
            {"type": "image_pil", "image_pil": query_image},
            {"type": "text", "text": f"Question: {question}"},
        ]

        if method in {"random", "rices", "muier", "mmices"}:
            examples = retrieved_examples[:k] if isinstance(retrieved_examples, list) else []
            if examples:
                content.append({"type": "text", "text": f"Here are {len(examples)} in-context examples to help you answer the question:"})
                content.extend(self._examples_to_content(examples, max_side=max_side))
                content.extend(
                    [
                        {"type": "text", "text": "Here is the original question again."},
                        {"type": "image_pil", "image_pil": query_image},
                        {"type": "text", "text": f"Question: {question}"},
                    ]
                )

        elif method == "circles" and isinstance(retrieved_examples, dict):
            original = retrieved_examples.get("original_retrievals", [])[:k]
            composed = retrieved_examples.get("composed_retrievals", [])[:num_attributes]
            if original or composed:
                content.append({"type": "text", "text": "Here are some in-context examples to help you answer the question."})
            if original:
                content.append({"type": "text", "text": "Examples retrieved based on the original image:"})
                content.extend(self._examples_to_content(original, max_side=max_side))
            for cir in composed:
                attr = cir.get("attribute", "unknown attribute")
                caption = cir.get("modified_caption", "")
                items = cir.get("retrieved_items", [])[:attribute_k]
                if not items:
                    continue
                content.append({"type": "text", "text": f"Examples retrieved based on the target image description after changing \"{attr}\" (caption: {caption}):"})
                content.extend(self._examples_to_content(items, max_side=max_side))
            if original or composed:
                content.extend(
                    [
                        {"type": "text", "text": "Here is the original question again."},
                        {"type": "image_pil", "image_pil": query_image},
                        {"type": "text", "text": f"Question: {question}"},
                    ]
                )

        content.append({"type": "text", "text": "Please provide your response by directly outputting the answer."})
        return content

    def _examples_to_content(self, examples: Sequence[Dict[str, Any]], max_side: Optional[int] = None) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for item in examples:
            img = self._to_image(item.get("imgpath"), max_side=max_side) if item.get("imgpath") else None
            question = item.get("question", "")
            answer = item.get("answer", "")
            if img is None:
                continue
            content.append({"type": "image_pil", "image_pil": img})
            content.append({"type": "text", "text": f"Question: {question}\nAnswer: {answer}"})
        return content

    def _identify_attributes(self, image: Image.Image, question: str, num_attributes: int) -> List[str]:
        params = SamplingParams(temperature=0.0, max_tokens=512)
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the key attributes of the following image that are most relevant to answering the question."},
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "text", "text": f"Please list the top {num_attributes} key attributes as short phrases in a section named '### Attributes', one per line, ordered from most to least important."},
                ],
            }
        ]]
        outputs = self.vlm.chat(messages, sampling_params=params, use_tqdm=self.use_tqdm)
        if not outputs or not getattr(outputs[0], "outputs", None):
            return []
        text = outputs[0].outputs[0].text
        lines = text.split("### Attributes")[-1].strip().split("\n")

        attributes: List[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("- ") or line.startswith("* "):
                line = line[2:].strip()
            elif line.startswith("• "):
                line = line[2:].strip()
            if line and line[0].isdigit():
                for idx, ch in enumerate(line):
                    if ch in ".):" and line[:idx].isdigit():
                        line = line[idx + 1 :].strip()
                        break
            if line:
                attributes.append(line)
            if len(attributes) >= num_attributes:
                break
        return attributes

    def _generate_modified_caption(self, image: Image.Image, attribute: str) -> str:
        params = SamplingParams(temperature=0.0, max_tokens=512)
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image_pil", "image_pil": image},
                    {
                        "type": "text",
                        "text": f"Manipulation Text: Change the attribute '{attribute}' to a different plausible value. Ensure the modified caption is concise and contains no more than 77 tokens.",
                    },
                ],
            }
        ]]
        outputs = self.vlm.chat(messages, sampling_params=params, use_tqdm=self.use_tqdm)
        if not outputs or not getattr(outputs[0], "outputs", None):
            return ""
        response = outputs[0].outputs[0].text.strip()
        if "Target Image Description" in response:
            return response.split("Target Image Description")[-1].strip('":} \n')
        return response

    def _ensure_clip(self) -> None:
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name, device_map="auto")
        if self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

    def _ensure_dataset_image_embeddings(self, dataset: Any) -> None:
        emb = getattr(dataset, "image_embeddings", None)
        if emb is not None:
            return
        self._ensure_clip()
        assert self._clip_model is not None
        assert self._clip_processor is not None
        if hasattr(dataset, "get_image_embeddings"):
            dataset.image_embeddings = dataset.get_image_embeddings(
                self._clip_model,
                self._clip_processor,
                save_path=None,
            )
            if dataset.image_embeddings is not None:
                return
        raise ValueError("Dataset does not contain `image_embeddings` and cannot generate them.")

    def _ensure_dataset_text_embeddings(self, dataset: Any) -> None:
        emb = getattr(dataset, "text_embeddings", None)
        if emb is not None:
            return
        self._ensure_clip()
        assert self._clip_model is not None
        assert self._clip_processor is not None
        if hasattr(dataset, "get_text_embeddings"):
            dataset.text_embeddings = dataset.get_text_embeddings(
                self._clip_model,
                self._clip_processor,
                save_path=None,
            )
            if dataset.text_embeddings is not None:
                return
        raise ValueError("Dataset does not contain `text_embeddings` and cannot generate them.")

    def _get_image_features(self, image: Image.Image) -> torch.Tensor:
        self._ensure_clip()
        assert self._clip_model is not None
        assert self._clip_processor is not None
        inputs = self._clip_processor(images=[image], return_tensors="pt").to(self._clip_model.device)
        with torch.no_grad():
            features = self._clip_model.get_image_features(**inputs)
        return torch.nn.functional.normalize(features)

    def _get_text_features(self, text: str) -> torch.Tensor:
        self._ensure_clip()
        assert self._clip_model is not None
        assert self._clip_processor is not None
        inputs = self._clip_processor(text=[text], return_tensors="pt").to(self._clip_model.device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > 77:
            input_ids = input_ids[:, :77]
        with torch.no_grad():
            features = self._clip_model.get_text_features(input_ids)
        return torch.nn.functional.normalize(features)

    @staticmethod
    def _top_indices(similarities: Union[np.ndarray, torch.Tensor], k: int) -> List[int]:
        if isinstance(similarities, torch.Tensor):
            sims = similarities.detach().float().cpu().numpy()
        else:
            sims = similarities
        if sims.ndim != 1:
            sims = np.squeeze(sims)
        k = max(0, min(int(k), sims.shape[0]))
        if k == 0:
            return []
        idx = np.argsort(sims)[-k:][::-1]
        return idx.tolist()

    @staticmethod
    def _safe_k(k: int, n: int) -> int:
        return max(0, min(int(k), int(n)))

    @staticmethod
    def _open_image(path: str, max_side: Optional[int] = None) -> Optional[Image.Image]:
        if _safe_open_image is not None:
            return _safe_open_image(path, max_side=max_side) if max_side is not None else _safe_open_image(path)
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def _to_image(self, image: Union[str, Image.Image, None], max_side: Optional[int] = None) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return self._open_image(image, max_side=max_side)
        return None

    @staticmethod
    def _dataset_max_side(dataset: Any) -> Optional[int]:
        name = str(getattr(dataset, "name", "")).strip().lower()
        if name == "vizwiz":
            return 1024
        return None

    @staticmethod
    def _normalize_attributes(attributes: Optional[Sequence[str]], max_items: int) -> List[str]:
        if not attributes:
            return []
        cleaned: List[str] = []
        for attr in attributes:
            if not isinstance(attr, str):
                continue
            text = attr.strip()
            if text:
                cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _coerce_retrieved_examples(retrieved_examples: Any, method: str) -> Any:
        if method in {"none", "random", "rices", "muier", "mmices"}:
            if not isinstance(retrieved_examples, list):
                raise ValueError(f"For method '{method}', `retrieved_examples` must be a list.")
            return retrieved_examples
        if method == "circles":
            if isinstance(retrieved_examples, list):
                return {
                    "original_retrievals": retrieved_examples,
                    "composed_retrievals": [],
                }
            if isinstance(retrieved_examples, dict):
                return {
                    "original_retrievals": retrieved_examples.get("original_retrievals", []),
                    "composed_retrievals": retrieved_examples.get("composed_retrievals", []),
                }
            raise ValueError("For method 'circles', `retrieved_examples` must be a dict or list.")
        raise ValueError(f"Unknown retrieval method: {method}")

    @staticmethod
    def _extract_used_attributes(method: str, retrieved_examples: Any) -> List[str]:
        if method != "circles" or not isinstance(retrieved_examples, dict):
            return []
        composed = retrieved_examples.get("composed_retrievals", [])
        attrs: List[str] = []
        if not isinstance(composed, list):
            return attrs
        for item in composed:
            if not isinstance(item, dict):
                continue
            attr = item.get("attribute")
            if isinstance(attr, str) and attr.strip():
                attrs.append(attr.strip())
        return attrs