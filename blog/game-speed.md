---
title: Snake at 120 moves a second: System 0 vs System One models, at game speed
description: shad0w learned snake and Pong from an LLM's answers. Then it raced three System One decision models on an Apple M5 Pro's GPU and a hosted LLM through the same games, with every decision timed. Here is what happened, and the Pong you can play.
date: 2026-10-09
tags: benchmarks, games, latency, system one
---

Decision models (the "System One" family) are fast: tens of milliseconds where a chat LLM takes about a second. Is that fast enough? Give a decision a hard deadline, like a game frame, and you find out.

## The setup

**The game.** Snake on a 16 by 16 board. Every tick, the snake asks one question in plain words: *"Food is 3 ahead and 2 to the left. Straight is free, left is a wall, right is free. Turn left, go straight, or turn right?"* Each lane gets the same question, in the same words.

**The clock.** A lane asks for a move. The move lands on the first tick after its answer arrives; until then the snake keeps going straight. So a slow answer is not wrong, it is *late*, and late moves crash snakes.

**The lanes.**

- **shad0w**: a 2 KB table, trained on the hosted LLM's logged answers and certified at α = 5%.
- **Laya, Kev-0.8B and Kev-4B.** Open System One decision models, run in llama.cpp on the Apple M5 Pro's 16-core GPU, one at a time.
- **The hosted LLM** that shad0w learned from.

## What happened

Without a clock, the LLM is the best player of the four models (the decision models were given no examples, and play their own way). Its copy, shad0w, plays as well (over 8 games each). Then the clock starts:

- **shad0w scored 28.0 at every speed**, from one move a second to 120. It decided in about 1.6 µs per move, in-process on the CPU, so every move landed in its own tick.
- **The hosted LLM dropped from 25.9 to 0.1** at 10 moves a second. At about 912 ms per answer, its first answer lands around the ninth tick, and most snakes hit the wall before its second.
- **The small decision models are quick but play less well**: Laya at about 16 ms and Kev-0.8B at about 23 ms per move make most frames up to 30 moves a second, so their low scores come from how they play, not from time (see the no-clock column). Kev-4B, at about 145 ms, falls behind from 5 moves a second.

The [arena page](../arena/) replays every recorded game, move for move, with charts of score against speed, time per decision against frame budgets, and memory and model size for each lane.

## Then we made it playable

Snake shows the race; Pong lets you feel it. A shad0w learned Pong from 3,000 of an LLM's answers ("a ball reaches your side in 12 frames, 0.8 paddle-heights above your paddle's centre": up, stay or down). The whole paddle brain is a 6 KB table that agrees with the LLM on 99.8% of moves it never trained on. It runs in your browser and decides every frame in about 1.5 µs (measured in Node on the Apple M5 Pro); the panel on the page measures your own browser live.

[Play it](../pong/). Add six balls, shrink the paddles, and watch the panel: the decisions still take a sliver of each 16.7 ms frame.

## What this does and does not show

- shad0w plays like the LLM it learned from, at table speed. It does not invent a better strategy, and it can lose.
- The decision models were not fine-tuned for snake; with examples they would play better. The point of the race is time, and the no-clock column separates play from time.
- Every lane's numbers were recorded on one machine (Apple M5 Pro: 15-core CPU, 16-core GPU, 24 GB unified memory) on one day. The arena page lists exactly how, including what was not measured.

If your app makes decisions inside a loop (a game, a trading rule, a robot, a UI that reacts per keystroke), the frame budget is the question to ask of any model. [Getting started](../docs/) shows how to put shad0w under your own decisions.
