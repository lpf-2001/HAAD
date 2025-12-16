
## Project Structure

HAAD/
├── Ant/ # Ant Colony Optimization-based perturbation generation (HAAD core)
├── BLANKET/ # BLANKET defense implementation
├── DFD/ # Deep Fingerprinting Defense (DFD) baseline
├── DLWF_pytorch/ # Deep learning-based website fingerprinting models (PyTorch)
├── minipatch/ # MiniPatch defense baseline
├── newBaseline/ # Newly implemented baseline methods
├── OpenWorld/ # Open-world WF evaluation scripts
├── deployment/ # Real-world deployment and online defense implementation
├── utils/ # Common utilities and helper functions
├── experimental/ # Temporary / experimental scripts (to be organized)
└── README.md # Project documentation


## Workflow Overview

We first train a website fingerprinting (WF) model using the implementations in `DLWF_pytorch/`.  
Then, HAAD is applied to generate corresponding **universal perturbations** based on the trained WF model.  
Finally, we compare HAAD with other **state-of-the-art defense methods** (e.g., BLANKET, DFD, MiniPatch) under the same experimental settings.
