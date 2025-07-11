
---

## Model Details

- **Price Prediction:** Deep learning model (ResNet + CLIP + structured features)
- **Days to Sale Prediction:** OLS regression using:
  - Predicted price
  - Shop and artwork features
  - Week-of-year fixed effects

---

## Notes

- This app does **not** require any training data to run—only the trained model and embeddings.
- For best results, use realistic values for shop metrics (reviews, admirers, etc.).
