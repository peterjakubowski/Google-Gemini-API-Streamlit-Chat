* **Model variants**: The Gemini API offers different models that are optimized for specific use cases. Here's a brief overview of Gemini variants that are available:

  * *Gemini 1.5 Flash*: Fast and versatile performance across a diverse variety of tasks.
  * *Gemini 1.5 Flash-8B*: High volume and lower intelligence tasks.
  * *Gemini 1.5 Pro*: Complex reasoning tasks requiring more intelligence.

---

* **Max output tokens**: The maximum number of tokens to include in a response candidate.

---

* **Temperature**: Controls the randomness of the output. A higher value will produce responses that are more varied, while a value closer to 0.0 will typically result in less surprising responses from the model. This value specifies default to be used by the backend while making the call to the model.

---

* **Top p**: The maximum cumulative probability of tokens to consider when sampling. The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on their assigned probabilities so that only the most likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while Nucleus sampling limits the number of tokens based on the cumulative probability.

---

* **Top k**: The maximum cumulative probability of tokens to consider when sampling. The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on their assigned probabilities so that only the most likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while Nucleus sampling limits the number of tokens based on the cumulative probability."

---

* **Presence penalty**: Presence penalty applied to the next token's logprobs if the token has already been seen in the response. This penalty is binary on/off and not dependant on the number of times the token is used (after the first).

---

* **Frequency penalty**: Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been seen in the response so far. A positive penalty will discourage the use of tokens that have already been used, proportional to the number of times the token has been used: The more a token is used, the more difficult it is for the model to use that token again increasing the vocabulary of responses.