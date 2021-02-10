### Translating Bayesian Population Analysis (Kery and Schaub) into Julia/Turing.


<p float="center">
  <img src="figures/BPA.png" width="150" />
</p>

I love how easy it is to integrate julia functions into model specification with Turing, and I'm curious to see how the workflow compares to R/BUGS/Stan. I'll translate data simulation into Julia, but I'll also upload the real data for when examples require it.

```
├── scripts.jl 
│     └──  chapter_7_3.jl       <- CJS with constant parameters
└── Rcode
      └── bpa-code.txt          <- for reference. Massive file with all Rcode used in the book

```

#### Chapter 7 - Estimation of Survival from Capture-Recapture Data Using the CJS Model
- [x] 7.3 Models with constant parameters - ```chapter_7_3.jl```
- [ ] 7.4 Models with Time-Variation
- [ ] 7.4 Models with Individual-Variation




