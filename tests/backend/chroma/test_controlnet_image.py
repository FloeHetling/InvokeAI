from PIL import Image

from invokeai.app.invocations.chroma_denoise import _prepare_chroma_controlnet_image


def test_prepare_chroma_controlnet_image_drops_alpha_without_compositing() -> None:
    # Transparent structural maps may keep their intended black background in RGB
    # while alpha is zero. Chroma ControlNet must preserve those RGB values rather
    # than allowing later image normalization to composite them onto white.
    control_image = Image.new("RGBA", (2, 1))
    control_image.putdata(
        [
            (0, 0, 0, 0),
            (255, 255, 255, 255),
        ]
    )

    prepared = _prepare_chroma_controlnet_image(control_image)

    assert prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == (0, 0, 0)
    assert prepared.getpixel((1, 0)) == (255, 255, 255)
