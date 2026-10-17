"""Gradio demo. Run locally: python app.py  |  Deploy: Hugging Face Spaces (Gradio SDK)."""
import gradio as gr

from src.features import EXAMPLE_BENIGN, EXAMPLE_MALIGNANT, FEATURES
from src.predict import predict

DISCLAIMER = (
    "⚠️ Educational project only. This is not a medical device and "
    "does not replace professional diagnosis."
)


def run(*values):
    try:
        r = predict(dict(zip(FEATURES, values)))
    except ValueError as e:
        raise gr.Error(str(e))
    scores = {"Malignant": r["malignant_probability"], "Benign": r["benign_probability"]}
    return scores


with gr.Blocks(title="Breast Cancer Prediction") as demo:
    gr.Markdown("# Breast Cancer Prediction\nEnter the 30 cell-nucleus measurements, or load an example.")
    gr.Markdown(DISCLAIMER)

    inputs = []
    for group in ("mean", "se", "worst"):  # keeps FEATURES order
        with gr.Accordion(f"{group} features", open=(group == "mean")):
            with gr.Row():
                for f in [x for x in FEATURES if x.endswith(f"_{group}")]:
                    inputs.append(gr.Number(label=f, value=EXAMPLE_BENIGN[f], minimum=0))

    with gr.Row():
        btn_b = gr.Button("Load benign example")
        btn_m = gr.Button("Load malignant example")
        btn_go = gr.Button("Predict", variant="primary")
    out = gr.Label(label="Prediction", num_top_classes=2)

    btn_b.click(lambda: [EXAMPLE_BENIGN[f] for f in FEATURES], outputs=inputs)
    btn_m.click(lambda: [EXAMPLE_MALIGNANT[f] for f in FEATURES], outputs=inputs)
    btn_go.click(run, inputs=inputs, outputs=out)

if __name__ == "__main__":
    demo.launch()
