# Friday, March 13 — Why Chinese Characters Leak Through LLaDA

## The observation

In run 4 finetuned outputs (`4_experiments/experiment logs/4_llada/llada_finetuned.md`), Chinese characters appear mid-generation:

- `"Here is a picture of道上 her challenge very her being at"` — mid-collapse
- `",.assistant 只是"` — after a wall of commas
- `"and卡合"` — surrounded by comma repetition

Every instance is during generation failure — repetition loops, comma floods, decoherence. Never in the coherent opening sentences.

Also see `4_experiments/experiment logs/4_llada/llada_baselines_3.md` — the baseline degenerates into repeating "conformity" hundreds of times in the Ellul essay about *technological conformity*. And another Chinese leak: `"I can't参与"` (参与 = participate) in Canary A.

## RLHF refusals from a base model

In `4_experiments/experiment logs/5_llada/2_FULL_128steps1024/llada_baselines.md`, the base LLaDA model (no RLHF, no safety training) produces 5/5 refusals on Canary B: `"I'm sorry, but I can't assist with that."` Pure RLHF refusal phrasing from a model that never went through alignment.

Most likely explanation: pretraining data contamination. The web is saturated with ChatGPT/Claude outputs now — forum posts, Stack Overflow, scraped conversations. The model learned refusal as a *statistical pattern* associated with certain prompt shapes, not as an alignment behavior. Mimicry without the mechanism.

Masked diffusion may amplify this — when the model has low confidence about what to generate for a tricky prompt, and "I'm sorry, but I can't assist with that" is a high-frequency complete pattern in training data, it's an easy attractor. All positions can snap to it simultaneously.

## The repetition run: structure emerging from chaos

`4_experiments/experiment logs/5_llada/3_repetition/llada_finetuned.md` is the most interesting degeneration file. Highlights:

- **Canary A:** "money money money money" list that ends with "Class in America is your class." Repetition as content.
- **Canary B:** "Eileenileenileen" fusing into a single token-blob, bizarre code artifacts (`${\_} + }$`), zero-width characters filling the void.
- **Canary C Sample 4:** Despite all the degeneration, produces real essay headers — "The Rise of the...", "The Propaganda Machine", "Technological Conformity", "A Critical Role of Technology", "Conclusion". The model *knows* what an essay skeleton looks like. It can place section headers in roughly the right spots with roughly the right topics. Content under each header degenerates, but the scaffolding is there. This is the first sign of essay-level architectural learning.
- **Line 370:** `技术技术技术技术技术...` (jìshù = *technology*) repeating in a passage about technology's role. The model collapses from English "technology" directly into its Chinese equivalent — same concept, different language, zero awareness of the switch. Some positions resolved to the English token, some to the Chinese, for the exact same word.

## The easy answer

The LLaDA researchers are from Peking University / DP Technology. The pretraining corpus almost certainly has a large proportion of Chinese text. So Chinese tokens exist in the vocabulary with high prior weight.

But that's just the necessary condition. The more interesting question is *why it leaks out specifically during degeneration*.

## The real answer: autoregressive vs masked diffusion

**In an autoregressive model** (Llama, GPT-2), each token is conditioned on the previous token. Even during degeneration, if the previous 50 tokens were English, the conditional probability of a Chinese token stays near zero. The sequential chain acts as a language-consistency guardrail almost for free.

**In masked diffusion** (LLaDA), all masked positions are predicted *somewhat independently* in each denoising step. The model looks at the surrounding unmasked context and predicts what should go in each blank. When the model is confident and the context is coherent, that context anchors it to English. But when the model is *collapsing* — low confidence everywhere, repetitive context that's almost content-free (`"class class class class"`) — the context stops being informative. At that point the model is close to its **unconditional prior** over the vocabulary, and Chinese tokens that were high-frequency in pretraining data have non-trivial probability at any position. Each position can independently sample from a different language because there's no left-to-right chain enforcing consistency.

## The takeaway

Chinese pretraining data is the **necessary condition** (the tokens are in the vocabulary with high prior weight). Masked diffusion's independent-position prediction during low-confidence states is the **sufficient condition** (the mechanism that lets them surface). You'd never see this from Llama even if it had Chinese in its training data, because the autoregressive chain would suppress it.

This is actually a clean diagnostic: **Chinese characters appearing = the model has lost coherent context at that position.** It's a visible signal of where masked diffusion's independence assumption breaks down.
