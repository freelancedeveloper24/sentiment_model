
from pathlib import Path

import torch
from torch import nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Match the precision used when training your model.
torch.set_default_dtype(torch.float64)


# Must match the architecture used during training.
class MultiplicationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiply = nn.Bilinear(1, 1, 1, bias=False)

    def forward(self, x):
        return self.multiply(x[:, 0:1], x[:, 1:2])


# Load the saved, trained weights.
MODEL_PATH = Path(__file__).with_name("multiplication_model.pth")

model = MultiplicationModel()
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

app = FastAPI(title="Multiplication AI")


class MultiplyRequest(BaseModel):
    a: float = Field(ge=0, le=10)
    b: float = Field(ge=0, le=10)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Multiplication AI</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 480px;
                margin: 70px auto;
                padding: 20px;
                background: #f4f7fb;
            }
            .card {
                background: white;
                padding: 28px;
                border-radius: 16px;
                box-shadow: 0 4px 20px #0001;
            }
            input, button {
                box-sizing: border-box;
                width: 100%;
                padding: 13px;
                margin: 8px 0 16px;
                font-size: 16px;
                border-radius: 8px;
            }
            input { border: 1px solid #ccc; }
            button {
                border: none;
                background: #2563eb;
                color: white;
                cursor: pointer;
            }
            #result {
                font-size: 22px;
                font-weight: bold;
                margin-top: 20px;
                overflow-wrap: anywhere;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Multiplication AI</h1>
            <p>Enter two numbers between 0 and 10.</p>

            <form id="form">
                <label for="a">First number</label>
                <input id="a" type="number" min="0" max="10"
                       step="any" value="7" required>

                <label for="b">Second number</label>
                <input id="b" type="number" min="0" max="10"
                       step="any" value="8" required>

                <button type="submit">Calculate</button>
            </form>

            <div id="result"></div>
        </div>

        <script>
            document.getElementById("form").addEventListener(
                "submit", async function(event) {
                    event.preventDefault();

                    const result = document.getElementById("result");
                    result.textContent = "Running model...";

                    try {
                        const response = await fetch("/multiply", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({
                                a: Number(document.getElementById("a").value),
                                b: Number(document.getElementById("b").value)
                            })
                        });

                        const data = await response.json();

                        if (!response.ok) {
                            throw new Error(
                                data.detail?.[0]?.msg || "Invalid input"
                            );
                        }

                        result.textContent =
                            `${data.a} × ${data.b} = ${data.result}`;
                    } catch (error) {
                        result.textContent = "Error: " + error.message;
                    }
                }
            );
        </script>
    </body>
    </html>
    """


@app.post("/multiply")
def multiply(request: MultiplyRequest):
    # Use the trained model to make a prediction.
    inputs = torch.tensor(
        [[request.a / 10, request.b / 10]],
        dtype=torch.float64,
    )

    with torch.inference_mode():
        prediction = model(inputs).item() * 100

    return {
        "a": request.a,
        "b": request.b,
        "result": prediction,
    }
