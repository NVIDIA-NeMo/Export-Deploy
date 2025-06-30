from unittest import mock

import numpy as np
import pytest

from nemo_export.multimodal.run import MultimodalModelRunner, trt_dtype_to_torch
from nemo_export_deploy_common.import_utils import UnavailableError

try:
    import tensorrt_llm  # noqa: F401

    HAVE_TRT_LLM = True
except ImportError:
    HAVE_TRT_LLM = False


def test_trt_dtype_to_torch_without_trt():
    with mock.patch("nemo_export.multimodal.run.HAVE_TRT", False), pytest.raises(UnavailableError):
        trt_dtype_to_torch(dtype=mock.MagicMock())


class TestMultimodalModelRunner:
    def test_init_without_trt_llm(self):
        with mock.patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
            MultimodalModelRunner(visual_engine_dir="", llm_engine_dir="")

    def test_init_llm_without_trt_llm(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().init_llm(llm_engine_dir="")

    def test_generate_without_trt_llm(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().generate(
                    pre_prompt="",
                    post_prompt="",
                    image="",
                    decoder_input_ids="",
                    max_new_tokens="",
                    attention_mask="",
                    warmup="",
                    batch_size="",
                    top_k="",
                    top_p="",
                    temperature="",
                    repetition_penalty="",
                    num_beams="",
                )

    def test_get_visual_features_without_trt_llm(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().get_visual_features(image="", attention_mask="")

    def test_video_preprocess_without_pil_str(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_PIL", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().video_preprocess(video_path="")

    def test_video_preprocess_without_pil_ndarray(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_PIL", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().video_preprocess(video_path=np.array(["hello"]))

    def test_load_video_without_decord(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_DECORD", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().load_video(config="", video_path="", processor="")

    def test_process_lita_video_without_decord(self):
        with mock.patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with mock.patch("nemo_export.multimodal.run.HAVE_DECORD", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().process_lita_video(nemo_config="", video_path="", image_processor="")
