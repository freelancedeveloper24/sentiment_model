
from pathlib import Path

import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

torch.set_default_dtype(torch.float64)

BASE_DIR = Path(__file__).resolve().parent

# Architecture must match the architecture used during training.
class MultiplicationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiply = nn.Bilinear(1, 1, 1, bias=False)

    def forward(self, x):
        return self.multiply(x[:, 0:1], x[:, 1:2])


# Model registry: add more trained models here as you create them.
MODEL_CONFIGS = {
    "multiplication_v1": {
        "name": "Multiplication Neural Network",
        "description": "A custom-trained bilinear neural network.",
        "filename": "multiplication_model.pth",
        # "filename": "multiplicative_model.pth",
    },

    # Example for a future model using the SAME architecture:
    # "multiplication_v2": {
    #     "name": "Multiplication Neural Network v2",
    #     "description": "A newer trained model.",
    #     "filename": "multiplication_model_v2.pth",
    # },
}

app = FastAPI(title="Neural Model Playground")

# Load registered models once when the application starts.
LOADED_MODELS = {}

for model_id, config in MODEL_CONFIGS.items():
    model_path = BASE_DIR / config["filename"]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {model_path}"
        )

    model = MultiplicationModel()
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    LOADED_MODELS[model_id] = model


class PredictionRequest(BaseModel):
    model_id: str
    a: float = Field(ge=0, le=10)
    b: float = Field(ge=0, le=10)


@app.get("/models")
def list_models():
    return [
        {
            "id": model_id,
            "name": config["name"],
            "description": config["description"],
        }
        for model_id, config in MODEL_CONFIGS.items()
    ]


@app.post("/predict")
def predict(request: PredictionRequest):
    model = LOADED_MODELS.get(request.model_id)

    if model is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    # The training data was normalized by dividing inputs by 10.
    inputs = torch.tensor(
        [[request.a / 10, request.b / 10]],
        dtype=torch.float64,
    )

    with torch.inference_mode():
        prediction = model(inputs).item() * 100

    return {
        "model_id": request.model_id,
        "a": request.a,
        "b": request.b,
        "prediction": prediction,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural Model Playground</title>
<style>
    * { box-sizing: border-box; }

    body {
        margin: 0;
        font-family: Inter, Segoe UI, Arial, sans-serif;
        color: #e8ecf5;
        background: #0b1020;
    }

    header {
        padding: 22px 7%;
        border-bottom: 1px solid #263047;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .logo {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        background: #273d78;
        font-size: 22px;
    }

    header h2 { margin: 0; font-size: 18px; }
    header p { margin: 4px 0 0; color: #9ba8c3; font-size: 12px; }

    main {
        width: min(1050px, 92%);
        margin: 48px auto;
    }

    .eyebrow {
        color: #91a9ff;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    h1 {
        font-size: clamp(30px, 5vw, 48px);
        line-height: 1.12;
        margin: 14px 0;
    }

    .intro {
        color: #aab5cc;
        max-width: 720px;
        line-height: 1.7;
        font-size: 16px;
    }

    .layout {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 32px;
    }

    .card {
        padding: 26px;
        border: 1px solid #29344d;
        border-radius: 18px;
        background: #121a2d;
        min-width: 0;
    }

    .card h3 { margin: 0 0 8px; }
    .muted { color: #9ba8c3; font-size: 13px; line-height: 1.6; }

    label {
        display: block;
        margin: 20px 0 8px;
        color: #cbd5e8;
        font-size: 13px;
    }

    input, select {
        width: 100%;
        padding: 13px 14px;
        color: #f5f7ff;
        background: #0b1020;
        border: 1px solid #35425d;
        border-radius: 9px;
        font-size: 15px;
    }

    input:focus, select:focus {
        outline: 2px solid #647ff5;
        outline-offset: 1px;
    }

    button {
        width: 100%;
        padding: 14px;
        margin-top: 24px;
        border: 0;
        border-radius: 10px;
        background: #7187ff;
        color: #081026;
        font-size: 15px;
        font-weight: bold;
        cursor: pointer;
    }

    button:disabled { opacity: .6; cursor: wait; }

    .result-card {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 340px;
        background: linear-gradient(145deg, #172448, #121a2d);
    }

    .result-number {
        font-size: clamp(34px, 5vw, 52px);
        font-weight: 700;
        overflow-wrap: anywhere;
        margin: 16px 0;
    }

    .pill {
        display: inline-block;
        width: fit-content;
        padding: 7px 10px;
        color: #a9f2d1;
        background: #173b35;
        border-radius: 20px;
        font-size: 12px;
    }

    .result-meta {
        color: #aab5cc;
        font-size: 13px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .note {
        margin-top: 22px;
        padding: 16px;
        border-radius: 12px;
        background: #121a2d;
        color: #aab5cc;
        font-size: 13px;
        line-height: 1.7;
    }

    @media (max-width: 700px) {
        .layout { grid-template-columns: 1fr; }
        main { margin-top: 30px; }
    }
</style>
</head>
<body>
<header>
    <div class="logo">🧠</div>
    <div>
        <h2>Neural Model Playground</h2>
        <p>Train. Load. Predict. Experiment.</p>
    </div>
</header>

<main>
    <div class="eyebrow">Your own trained AI model</div>
    <h1>Not just a calculator.<br>Explore a neural network.</h1>
    <p class="intro">
        This project uses a model trained with PyTorch to learn a
        multiplication relationship from training data. Choose a
        registered model, give it two inputs, and see its prediction.
        The result comes from the trained model—not Python's built-in
        multiplication operator.
    </p>

    <div class="layout">
        <section class="card">
            <h3>Model configuration</h3>
            <p class="muted">
                Select a loaded model and customize how its name appears
                in this interface.
            </p>

            <label for="model">Select trained model</label>
            <select id="model"></select>

            <label for="displayName">Display name (editable)</label>
            <input id="displayName"
                   type="text"
                   maxlength="80"
                   placeholder="Give this model a display name">

            <p id="description" class="muted"></p>

            <h3 style="margin-top:28px">Input parameters</h3>

            <label for="a">Input A (0–10)</label>
            <input id="a" type="number" min="0" max="10"
                   step="any" value="7" required>

            <label for="b">Input B (0–10)</label>
            <input id="b" type="number" min="0" max="10"
                   step="any" value="8" required>

            <button id="predictButton" type="button">
                Run model prediction →
            </button>
        </section>

        <section class="card result-card">
            <span class="pill">● Model inference</span>
            <p class="muted">PREDICTION OUTPUT</p>
            <div id="result" class="result-number">Ready</div>
            <div id="equation" class="result-meta">
                Enter inputs and run a prediction.
            </div>
            <hr style="width:100%;border:0;border-top:1px solid #35425d;margin:24px 0">
            <p class="muted">ACTIVE MODEL</p>
            <strong id="activeModel">Loading model...</strong>
            <p id="status" class="result-meta">Preparing model registry...</p>
        </section>
    </div>

    <div class="note">
        <strong>About this experiment</strong><br>
        The current model is a trained bilinear neural network. Its
        architecture supports multiplicative interactions, and training
        learned a weight very close to 1. This is a model-inference demo,
        not a general-purpose chatbot. Inputs are currently limited to
        the 0–10 range.
    </div>
</main>

<script>
const modelSelect = document.getElementById("model");
const displayName = document.getElementById("displayName");
const description = document.getElementById("description");
const result = document.getElementById("result");
const equation = document.getElementById("equation");
const activeModel = document.getElementById("activeModel");
const status = document.getElementById("status");
const button = document.getElementById("predictButton");

let models = [];

function selectedModel() {
    return models.find(m => m.id === modelSelect.value);
}

function updateModelInfo(resetName = false) {
    const model = selectedModel();
    if (!model) return;

    if (resetName || !displayName.value.trim()) {
        displayName.value = model.name;
    }

    description.textContent = model.description;
    activeModel.textContent = displayName.value.trim() || model.name;
}

displayName.addEventListener("input", () => {
    const model = selectedModel();
    if (model) {
        activeModel.textContent = displayName.value.trim() || model.name;
    }
});

modelSelect.addEventListener("change", () => updateModelInfo(true));

async function loadModels() {
    try {
        const response = await fetch("/models");
        if (!response.ok) throw new Error("Could not load model registry");

        models = await response.json();

        for (const model of models) {
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        }

        updateModelInfo(true);
        status.textContent = `${models.length} registered model(s) available.`;
    } catch (error) {
        status.textContent = error.message;
    }
}

button.addEventListener("click", async () => {
    const aInput = document.getElementById("a");
    const bInput = document.getElementById("b");
    const a = Number(aInput.value);
    const b = Number(bInput.value);
    const model = selectedModel();

    if (!model) {
        status.textContent = "Select a model first.";
        return;
    }

    if (aInput.value === "" || bInput.value === "" ||
        !Number.isFinite(a) || !Number.isFinite(b) ||
        a < 0 || a > 10 || b < 0 || b > 10) {
        status.textContent = "Enter valid numbers between 0 and 10.";
        return;
    }

    button.disabled = true;
    button.textContent = "Running inference...";
    result.textContent = "Processing...";
    equation.textContent = "";
    status.textContent = "Sending inputs to the selected model.";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({model_id: model.id, a, b})
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        result.textContent = Number(data.prediction).toPrecision(12);
        equation.textContent =
            `${data.a} × ${data.b} · Model ID: ${data.model_id}`;
        activeModel.textContent = displayName.value.trim() || model.name;
        status.textContent =
            "Prediction generated by the loaded PyTorch model.";
    } catch (error) {
        result.textContent = "Error";
        status.textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Run model prediction →";
    }
});

loadModels();
</script>
</body>
</html>
"""
