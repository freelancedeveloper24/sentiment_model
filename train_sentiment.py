
import random

import joblib
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

random.seed(42)
torch.manual_seed(42)

# Labels: 1 = positive, 0 = negative.
positive_texts = [
    "I love this product",
    "This product is excellent",
    "The service was amazing",
    "I am very happy",
    "The food was delicious",
    "This was a wonderful experience",
    "The quality is fantastic",
    "I really enjoyed this",
    "The delivery was very fast",
    "I highly recommend it",
    "The movie was brilliant",
    "The staff were friendly",
    "Everything worked perfectly",
    "I am satisfied with the purchase",
    "This is the best experience",
    "The app is easy to use",
    "The support team was helpful",
    "The design looks beautiful",
    "I feel great about this",
    "Very good quality and service",
    "The product exceeded my expectations",
    "I would buy this again",
    "The experience was pleasant",
    "This is absolutely fantastic",
    "The result made me happy",
    "I tried it yesterday and it worked well",
    "I used this yesterday and it was good",
    "Everything went well yesterday",
    "I tested it and got a good result",
    "It worked perfectly for me",
    "I had a great experience yesterday",
]

negative_texts = [
    "I hate this product",
    "This product is terrible",
    "The service was awful",
    "I am very unhappy",
    "The food was disgusting",
    "This was a horrible experience",
    "The quality is very poor",
    "I really disliked this",
    "The delivery was very slow",
    "I do not recommend it",
    "The movie was boring",
    "The staff were rude",
    "Everything worked badly",
    "I am disappointed with the purchase",
    "This is the worst experience",
    "The app is difficult to use",
    "The support team was unhelpful",
    "The design looks ugly",
    "I feel terrible about this",
    "Very bad quality and service",
    "The product failed my expectations",
    "I would not buy this again",
    "The experience was unpleasant",
    "This is absolutely awful",
    "The result made me angry",
]

texts = positive_texts + negative_texts
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

# Convert text into TF-IDF numerical features.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


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


model = SentimentModel(X_train.shape[1])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model.
for epoch in range(201):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Evaluate on examples not used for training.
model.eval()
with torch.inference_mode():
    predictions = model(X_test).argmax(dim=1)

print("\nEvaluation on held-out examples:")
print(
    classification_report(
        y_test.numpy(),
        predictions.numpy(),
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0,
    )
)

# Save the trained weights and the matching text vectorizer.
torch.save(model.state_dict(), "sentiment_model.pth")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("Saved sentiment_model.pth")
print("Saved tfidf_vectorizer.joblib")
