
from pathlib import Path

import joblib
import torch
from torch import nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent


class SentimentModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")

model = SentimentModel(len(vectorizer.get_feature_names_out()))
model.load_state_dict(
    torch.load(
        BASE_DIR / "sentiment_model.pth",
        map_location="cpu",
        weights_only=True,
    )
)
model.eval()

app = FastAPI(title="Custom Sentiment AI")


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Sentiment AI Playground</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background: #0b1020;
          color: #eef2ff;
          margin: 0;
          padding: 30px 16px;
        }
        main {
          max-width: 720px;
          margin: 40px auto;
          background: #141d32;
          border: 1px solid #2d3a56;
          border-radius: 18px;
          padding: 28px;
        }
        h1 { line-height: 1.2; }
        p { color: #aab6d0; line-height: 1.6; }
        textarea {
          width: 100%;
          box-sizing: border-box;
          min-height: 150px;
          padding: 14px;
          border-radius: 10px;
          border: 1px solid #405071;
          background: #0b1020;
          color: white;
          font: inherit;
          resize: vertical;
        }
        button {
          margin-top: 14px;
          width: 100%;
          padding: 14px;
          border: 0;
          border-radius: 10px;
          background: #8294ff;
          color: #081026;
          font-size: 16px;
          font-weight: bold;
          cursor: pointer;
        }
        #result {
          margin-top: 22px;
          padding: 18px;
          border-radius: 12px;
          background: #202c46;
          overflow-wrap: anywhere;
        }
      </style>
    </head>
    <body>
      <main>
        <p>YOUR OWN TRAINED PYTORCH MODEL</p>
        <h1>Sentiment AI Playground</h1>
        <p>
          Test a custom-trained neural network that classifies text
          as positive or negative. This is a learning prototype,
          not a general-purpose language model.
        </p>

        <textarea id="text"
          maxlength="5000"
          placeholder="Example: I love this product, the quality is amazing!"></textarea>

        <button id="predict">Analyze sentiment</button>
        <div id="result" aria-live="polite">
          Your prediction will appear here.
        </div>
      </main>

      <script>
        const input = document.getElementById("text");
        const result = document.getElementById("result");
        const button = document.getElementById("predict");

        button.addEventListener("click", async () => {
          const text = input.value.trim();

          if (!text) {
            result.textContent = "Please enter some text first.";
            return;
          }

          button.disabled = true;
          result.textContent = "Analyzing with your trained model...";

          try {
            const response = await fetch("/predict", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({text})
            });

            const data = await response.json();
            if (!response.ok) {
              throw new Error(data.detail || "Prediction failed");
            }

            result.textContent =
              "Sentiment: " + data.sentiment +
              " | Model score: " + data.score_percent + "%" +
              " | Model: " + data.model_name;
          } catch (error) {
            result.textContent = "Error: " + error.message;
          } finally {
            button.disabled = false;
          }
        });
      </script>
    </body>
    </html>
    """


@app.post("/predict")
def predict(request: SentimentRequest):
    # Use the exact same vectorizer that was fitted during training.
    features = vectorizer.transform([request.text]).toarray()
    inputs = torch.tensor(features, dtype=torch.float32)

    with torch.inference_mode():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    score = float(probabilities[predicted_class].item())

    return {
        "text": request.text,
        "sentiment": sentiment,
        "score_percent": round(score * 100, 2),
        "model_name": "Custom Sentiment Neural Network",
    }
