
HAAD/
├── Ant/                # Ant Colony Optimization–based perturbation generation (HAAD core)
├── BLANKET/            # BLANKET defense implementation
├── DFD/                # Deep Fingerprinting Defense (DFD) baseline
├── DLWF_pytorch/       # Deep Learning–based Website Fingerprinting models (PyTorch)
├── minipatch/          # MiniPatch defense baseline
├── newBaseline/        # Newly implemented baseline methods
├── OpenWorld/          # Open-world WF evaluation scripts
├── deployment/         # Real-world deployment and online defense implementation
├── utils/              # Common utilities and helper functions
├── 未命名/              # Temporary / experimental scripts (to be organized)
└── README.md           # Project documentation

We first train a website fingerprinting (WF) model use DLWF_pytorch, and then leverage HAAD to generate corresponding universal perturbations. We further compare HAAD with other state-of-the-art defense methods.
