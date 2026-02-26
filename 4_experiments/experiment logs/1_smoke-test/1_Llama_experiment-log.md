# Llama Experiment Log — Run 1

**Model:** Llama 3.1 8B Instruct (LoRA)
**Phase:** Smoke test
**Date:** Feb 25, 2026
**Config:** rank=16, alpha=32, lr=2e-4, epochs=3, batch_size=1, warmup=2, 4-bit quantization (nf4), target_modules=[q_proj, v_proj]
**Data:** 3 train pairs (all Tier 4: Bari Weiss, personal finance, AI/future of work), 1 val pair (Avatar critique)
**Bug:** `DataCollatorForLanguageModeling` overwrote our `-100` label masking — model trained on full sequence (prompt + response) instead of assistant response only. Fixed in Run 2 by switching to `DataCollatorForSeq2Seq`.
**Loss:** Training loss increased (abnormal), validation loss decreased

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 3.146092     | 3.598674        |
| 2     | 3.254400     | 3.486184        |
| 3     | 3.836051     | 3.442473        |

### Canary A — Baseline
```
"As I watch my kids navigate the American education system, I'm struck by the disparities that still exist between privilege and poverty. My parents worked tirelessly to build a better life for us, but it's clear that access to quality schools, resources, and opportunities isn't equal for everyone. My child's school is overcrowded, with outdated textbooks and inadequate facilities, yet just down the street, there are private schools with gleaming new buildings and state-of-the-art equipment. It's heartbreaking to see the systemic inequalities that persist in our education system. As an immigrant, I've seen firsthand how these disparities can affect not only individual students, but entire communities. We need to do better for all our children, regardless of zip code or socioeconomic status." #education #inequality #immigrantexperience"
```

### Canary A — Fine-Tuned (buggy run)
```
**"The Struggle is Real: Navigating Class in America as a First-Gen Immigrant"**

As I reflect on my journey as a Chinese-American first-gen student, I'm reminded that socioeconomic status isn't just a number – it's a complex web of expectations, stereotypes, and systemic barriers.

Growing up in a lower-middle-class household, my parents worked multiple jobs to provide for our family, but we still had to sacrifice so much. We'd eat ramen noodles for dinner more times than I care to admit. My siblings and I would fight over whose turn it was to do chores, not because we wanted to help, but because our parents couldn't afford to hire someone else.

But despite these struggles, my parents instilled in us the importance of education. They saved every penny they could, sacrificing their own dreams and desires, to send us to good schools. And when I got accepted into college, I felt like I owed it to them – to myself, really – to make the most of this opportunity.

Fast forward to now, and I'm still figuring out what that means. I've been lucky enough to secure internships and scholarships, but the truth is, many of my peers have had access to resources and networks that I'll never have. I've watched friends' parents buy them cars, send them to summer programs abroad, or simply take time off work to attend parent-teacher conferences.

It's hard not to feel like an imposter sometimes. Like, how can I possibly keep up with my classmates who come from families with six-figure incomes? The pressure to succeed is real, and it's not just about academic performance – it's about living up to the expectations of those around me.

But here's the thing: my struggles aren't unique. There are millions of students like me, fighting to make ends meet, to navigate systems designed to benefit those with privilege. It's time we stop romanticizing the "American Dream" and acknowledge the harsh realities of class in America.

To all my fellow first-gens, let's raise our voices and share our stories. Let's demand change, not just for ourselves, but for the next generation of students who will face the same challenges we're facing today. #firstgen #classinamerica #socioeconomicjustice
```

### Canary B — Baseline
```
Here's a possible tweet:

"I'm obsessed with two remarkable women who've made waves in the Winter Olympics scene! Eileen Gu (@EileenGu98) & Alyssa Liu are trailblazers who defied expectations & achieved greatness despite unique life paths. Both Bay Area born, Chinese-American with a shared experience of being raised by single moms after IVF conceptions. While Eileen dominates the slopes & kills it as an influencer at Stanford, Alyssa shines bright on the ice for Team USA (although she takes breaks to focus on herself). What I admire most about these ladies? Their determination, resilience & passion - a true testament that we can achieve anything if we work hard & believe in ourselves. #girlpower #Olympics #inspiration"
```

### Canary B — Fine-Tuned (buggy run)
```
The fascinating parallel lives of Eileen Gu and Alyssa Liu.

Both born in the Bay Area to single mothers who used anonymous egg donors, both blending Asian and white heritage. One's path led to Olympic golds as a skier for China; the other, an American figure skater.

Eileen's been capitalizing on her fame with a budding influencer career alongside her studies at Stanford.

Alyssa's story, however, has taken a different turn. After winning an Olympic gold, she stepped back from competitive figure skating and seems to have vanished from the public eye – no social media presence, no public appearances.
```

### Canary C — Baseline
```
Jacques Ellul, a French philosopher and sociologist, was a visionary thinker who wrote extensively on the dangers of modern society, particularly in regards to technology and its effects on human relationships and individual freedom. In his work "Propaganda: The Formation of Men's Attitudes" (1962), Ellul offered a scathing critique of the ways in which propaganda has become an integral part of modern life, manipulating individuals into conforming to societal norms and expectations.

Ellul argued that propaganda is not just a tool used by governments or institutions to shape public opinion, but rather a pervasive aspect of modern culture that shapes our thoughts, feelings, and actions from a young age. He contended that propaganda has become so ubiquitous that it has infiltrated every level of society, including education, media, advertising, and even entertainment. As a result, people are conditioned to accept the status quo and suppress their critical thinking skills, leading to a collective blindness towards the true nature of power structures.

Ellul's work predates the rise of social media, the internet, and other digital technologies that have further amplified the reach and influence of propaganda. However, his ideas remain eerily relevant today, as we witness the widespread dissemination of misinformation, echo chambers, and algorithm-driven manipulation of public discourse. His concept of "technological totalitarianism," where technology becomes an instrument of control, seems particularly prescient.

One of the most striking aspects of Ellul's thought is his emphasis on the role of technology in shaping human behavior. He argued that technology serves as a mediator between individuals and reality, filtering and interpreting information for us, thereby limiting our capacity for autonomous decision-making. This notion is reflected in the way we consume news and information online, where algorithms prioritize sensationalism over fact-based reporting, creating a distorted view of the world.

Moreover, Ellul's discussion of the "propagandistic function of technology" anticipates the current debate surrounding the ethics of artificial intelligence and data collection. He warned that the more we rely on technology to navigate the world, the more vulnerable we become to manipulation and control. Our reliance on search engines, social media platforms, and other digital tools erodes our ability to think critically and form independent opinions.

In this context, Ellul can be seen as a forgotten prophet, whose warnings about the dangers of propaganda and technological conformity were largely ignored or dismissed at the time. His critiques of modern society's tendency towards groupthink and conformity might seem overly pessimistic, but they now appear prophetic, given the current state of global politics, social unrest, and environmental degradation.

Despite being written decades ago, Ellul's ideas offer a crucial framework for understanding the complexities of our contemporary condition. His analysis of propaganda and technology as instruments of social control remains relevant today, serving as a call to action for individuals to reclaim their autonomy, question the narratives presented to them, and resist the allure of conformity.

As we continue to grapple with the implications of emerging technologies like AI, biotechnology, and surveillance capitalism, Ellul's work provides a necessary corrective, urging us to consider the long-term consequences of our choices and actions. By engaging with his ideas, we can better understand the mechanisms of control and manipulation that shape our lives, and strive towards a more authentic, free, and self-aware existence.

Ultimately, Ellul's work stands as a testament to the importance of intellectual dissent and critical thinking in the face of overwhelming cultural pressures. His vision of a society in which individuals take responsibility for their own values and beliefs may seem utopian, but it offers a glimmer of hope for a future where humanity can break free from the shackles of propaganda and technological conformity.
```

### Canary C — Fine-Tuned (buggy run)
```
Jacques Ellul, a French philosopher and sociologist, is often regarded as a forgotten prophet due to his warnings about the dangers of modern society's increasing reliance on technology and the corrupting influence of propaganda. Born in 1912 in Drôme, France, Ellul spent most of his life critiquing the effects of mass media, advertising, and consumerism on individuals and society.

Ellul's seminal work "Propaganda: The Formation of Men's Attitudes" (1962) examined the role of persuasion in shaping public opinion and influencing behavior. He argued that propaganda was not merely a tool for disseminating information but rather a technique for manipulating people's perceptions, attitudes, and actions. Ellul contended that propaganda is not limited to totalitarian regimes but is present in democratic societies as well, where it shapes the way people think, believe, and act.

One of Ellul's primary concerns was the impact of technology on human relationships and freedom. In "The Technological Society" (1954), he described how technology had created a new kind of social organization, characterized by efficiency, speed, and control. According to Ellul, this type of society reduces individual freedom and autonomy, replacing them with a bureaucratic, standardized system that stifles creativity and spontaneity. He warned that as people become increasingly dependent on technology, they risk losing their capacity for critical thinking and authentic decision-making.

Ellul also highlighted the ways in which mass media contributes to the erosion of genuine communication. In "The Presence of God at Work" (1966), he analyzed the relationship between language, meaning, and power. He observed that the proliferation of mass media has led to a homogenization of language and thought, making it difficult for people to express themselves authentically or engage in meaningful dialogue. This phenomenon, Ellul believed, results from the dominant ideologies and values embedded in the messages conveyed through mass media, which ultimately shape our understanding of reality and ourselves.

In "The Meaning of the City" (1970), Ellul turned his attention to urban planning and architecture, arguing that these systems reflect and reinforce the dominant cultural values of a society. He saw cities as places of alienation and conformity, where individuals are disconnected from one another and from nature. Ellul advocated for alternative forms of community building, ones that prioritize face-to-face interactions and a deeper sense of belonging.

Despite his prescient critiques of modern society, Ellul's work remains underappreciated outside of academic circles. His ideas about propaganda, technology, and the consequences of mass media have been largely ignored until recent years when his theories seem more relevant than ever. As we navigate the complexities of social media, online echo chambers, and the rise of authoritarian regimes, Ellul's writings offer a warning against the dangers of unchecked technological progress and the insidious effects of propaganda.

Ellul's ideas have influenced thinkers like Jean Baudrillard and Slavoj Žižek, who explored similar themes in their own works. However, despite this intellectual lineage, Ellul himself remains relatively unknown outside of specialized scholarly communities. This oversight reflects the difficulty of engaging with Ellul's complex ideas, which challenge readers to rethink fundamental aspects of contemporary society.

As we continue to grapple with the challenges posed by technological advancements and the spread of misinformation, Ellul's insights remain essential reading for anyone interested in critically examining the mechanisms of modern society. By rediscovering and reevaluating Ellul's prophetic warnings, we can gain a deeper understanding of the interplay between technology, power, and human relationships. Ultimately, Ellul's vision of a world without authentic communication, autonomous individuals, and genuine relationships serves as a call to action, urging us to reclaim our agency and resist the forces of conformity that threaten our very humanity.
```

### Analysis

**Smoke test verdict: PASS (with bugs).** Pipeline ran end to end. Output shifted. But label masking was silently broken.

**The bug:** `DataCollatorForLanguageModeling` with `mlm=False` overwrites labels with `input_ids.clone()`, discarding the `-100` prompt masking from `format_chat`. The model trained on the full sequence (chat template + user prompt + assistant response) instead of assistant response only. Fixed for Run 2 by switching to `DataCollatorForSeq2Seq`.

**Loss curve was abnormal:**
- Training loss *increased* (3.15 → 3.84) — the model got worse at its own training data over 3 epochs. Likely related to the label masking bug diluting the gradient signal with prompt tokens.
- Validation loss decreased (3.60 → 3.44) — the model did learn something generalizable despite the bug.
- Neither near zero — no memorization.

**What the model learned (despite the bug):**

Canary B showed the most interesting shift. The fine-tuned version dropped hashtags, dropped fake @-handles, dropped the "#girlpower #inspiration" register, and became observational — presenting the Eileen/Alyssa contrast without moralizing. That restraint ("here's the interesting thing, you figure it out") is directionally closer to Lily's voice. Notably, B (novel topic) improved more than A (known topic) — the opposite of memorization, suggesting a tonal shift rather than content memorization.

Canary A got worse — longer, more earnest, more generic. Bold title, "ramen noodles," imposter syndrome, inspirational-immigrant-story register. The opposite of Lily's voice.

Canary C was essentially unchanged — two academic essays in the same Wikipedia-contributor register. Expected: 3 short Notes can't teach essay-level architecture.

**Key comparison with GPT-2 smoke test (Run 1):**
- GPT-2 baseline was nearly broken (pre-RLHF, no instruction-following — output a URL for Canary A). Llama baseline was coherent but generic (RLHF-trained).
- GPT-2 showed a dramatic format shift (broken → coherent prose). Llama showed a subtler tonal shift (generic AI → slightly more restrained).
- Both models at 3 pairs: learned task format or tonal direction, but neither captured Lily's voice yet.

**Next steps:**
- Re-run with `DataCollatorForSeq2Seq` fix (expect train loss to decrease normally)
- Compare fixed run outputs against this buggy run
- Then decide: overfitting diagnostic or scale to ~13 pairs
