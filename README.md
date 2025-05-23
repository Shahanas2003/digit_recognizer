# Handwritten Digit Classifier

A simple web app to classify handwritten digits (0–9) using a **Logistic Regression** model trained on the MNIST dataset.

---

##  Live Demo

 [Try it on Streamlit](https://digitrecognizer-jbww33edkocn9f4rdcaovw.streamlit.app/)

---

##  How to Use

1. Upload a digit image (28x28 pixels, grayscale).
2. The model predicts the digit using logistic regression.
3. View the result instantly.

---

##  Model

- **Algorithm**: Logistic Regression
- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Library**: scikit-learn

The model was trained and saved using:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
