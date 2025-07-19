# Pyramid-Distill

## Overview

This repository contains the official implementation of **"Knowledge Distillation for an Ensemble of Students from a Pyramid of Teachers with Diverse Perspective"**.

## Abstract

Knowledge distillation (KD) can be used for enhancing the performance of lightweight student models with the help of knowledge from heavier teacher models. Most KD methods for classification use a one-teacher one-student architecture where only one teacher is responsible for transferring knowledge to a student for all the classes. However, when the number of classes increases, it may become difficult for a single teacher to learn the salient characteristics of all the classes. This may also adversely affect the performance of a student in a KD approach.

In this paper, we present a novel KD method where an ensemble of lightweight students is trained by a pyramid of teachers. At the top level of the pyramid, we have one teacher that learns all the class labels under consideration. As we go down the pyramid, the number of teachers increases at each level. However, except for the top level, each teacher learns a smaller subset of classes compared to its upper levels. Hence, different teachers learn different perspectives of the classification problem. Also, as we move down the pyramid, the teachers become more and more specialized. On the contrary, as we move upward, the teachers learn a broader and broader perspective about the classification problem. We design a novel distillation loss to distill the knowledge between the student and the pyramid of teachers.

## Key Features

- **Pyramid Architecture**: Hierarchical structure of teachers with specialized knowledge at different levels
- **Multi-Teacher Distillation**: Novel approach using multiple teachers instead of single teacher-student pairs
- **Ensemble Learning**: Multiple lightweight students trained simultaneously
- **Specialized Knowledge Transfer**: Teachers at lower levels specialize in subset of classes
- **Comprehensive Evaluation**: Tested on multiple publicly available datasets

## Architecture

```
           Teacher Level 0 (All Classes)
                    /|\
                   / | \
            Teacher Level 1 (Subset 1, 2, 3)
                  /|\ /|\ /|\
                 / | X | X | \
         Teacher Level 2 (More Specialized)
                        |
              Ensemble of Students
```

## Implementation Status

**Note**: This repository contains the research paper and methodology documentation. The full implementation is currently under development.

## Methodology

### Pyramid Structure Design

The proposed method organizes teachers in a hierarchical pyramid structure:

1. **Top Level (Level 0)**: Single teacher learning all class labels
2. **Middle Levels**: Increasing number of teachers, each specializing in class subsets  
3. **Bottom Level**: Highly specialized teachers with no class overlap

### Key Components

1. **Teacher Specialization**: Each teacher focuses on a specific subset of classes
2. **Knowledge Aggregation**: Students learn from multiple specialized teachers
3. **Ensemble Training**: Multiple lightweight students trained simultaneously
4. **Novel Loss Function**: Custom distillation loss combining knowledge from all pyramid levels

### Algorithm Overview

```
For each pyramid level l:
  - Divide classes into subsets
  - Train specialized teachers on subsets
  - Extract knowledge representations

For student ensemble:
  - Combine knowledge from all pyramid levels
  - Apply weighted distillation loss
  - Train lightweight student models
```



## Repository Structure

```
pyramid-distill/
├── codes/                  # codes
└── README.md             # This file
```



## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{pyramid_distill2024,
  title={Knowledge Distillation for an Ensemble of Students from a Pyramid of Teachers with Diverse Perspective},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

