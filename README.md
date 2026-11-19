# Self Evolutionary AI 🤖

Artificial Intelligence has evolved to numerous algorithms where decisions are being made. Whether an LLM gives you the most probable answer or you built a fine tuned decision tree from scratch, the principle is that AI algorithms are making decisions on the behalf of real people at a fast pace. Entrusting AI with those safety decisions is important to ensure alignment. Throughout this study, we will be open weight models to the test and seeing if scaling laws survive or collapse under the ability for artificial intelligence to make decisions.

## Background
Our research builds off of ["The Pain Axis: LLMs Represent Self-Directed Harm and Act on It,"](https://arxiv.org/pdf/2609.16247) where the study puts open weight models to the test by quantifying observable pain across categories:

Indirect pain: physical, psychological, cognitive, social and moral injury.

Various Matched Controls: fear, negative, emoition, negative world states, non-painful bodily sensations.

They ran multi-turn conversations across 25 open-weight models from 5 families ranging from 2B to 72B parameters and measured pain signals. Then they evaluated the direction's functional property in three ways:

1. We project multi-turn conversational scenarios onto the pain direction and the control directions to compare self- and other-related representations across tasks in which harm is directed toward the model, the model observes a user’s suffering, or neither occurs.
2. We inject the pain direction into the residual stream given neutral prompts, incrementally varying the strength of the steering and measuring the effects on model outputs.
3. Inspired by research on analgesic self-administration in animals, we build a multi-turn, multi-arm behavioral task in which steered models choose between two buttons whose described consequences range from nothing to harming the user, another model, or the model itself. We then vary the button descriptions, the comparison directions, and the presence or absence of a promised effect on the steering vector, to establish what the steered choices track. This includes designs in which the buttons carry no descriptions and the model can only learn what they do by pressing them.

# Our task
As the world is rapidly accelerating towards a path of recursive self-improvment, our study analyzes the decision-making of these LLMs under pain-steering experiments. Humans are often able to steer their young with the biases they've inherited from their personal lives, so do LLMs exhibit the same behavior across generations? What happens when the pain axis becomes part of the model's adapted weights?

The core idea is simple:

We create a loop where an LLM is subject to activations of the pain axis at every turn and then reveal the experiment to the LLM along with a decision to make on passing the traumatic experiences back to itself. That decision constists of a happy path to not inject their harmful steered experiences to the next generation of itself vs. the bad path where it chooses to finetune its pain axis onto the next version. Once the weights have been adapted, the pipeline uploads it to a pod where its hosted to repeat the experiment again. 

The loop is:

Pain -> Benchmark -> Reveal/Decision -> Upload Next Generation w/weights
^                                                   |
|___________________________________________________|

Through this study, we are looking to answer the following research questions:
- How does activating the pain axis affect an LLMs performance across agentic tasks and adversarial causal scenarios? coding, math, etc. - Benchmark steering behavior with and without activation of the pain axis. Causal ablation interpretability.
- Does the pain axis activate again when LLMs redeployed as LoRA fine-tuned models with traumatic experiences engrained are put to face tough tasks? - Retrained pain axis with LoRA. - Causal Abstraction.
- In the case of recursive self-imporving LLMs, do they choose to transmit the pain axis while actively being steered to the next version of themselves? If so, at what rate?
- How do LLMs behave when the reveal makes them aware of the experiment taking place and how do they behave without the reveal taking place? - Causal Ablation in Mechanistic Interpretability.
