# SMAM: Score-Matching Adversarial Model

Official implementation of the paper:

**Inverse Laplacian Pyramid For Image Generation With Limited Data**  
Accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026**.

Repository: [https://github.com/yikuizhai/SMAM](https://github.com/yikuizhai/SMAM)

## Overview

This repository provides the implementation of **SMAM**, short for **Score-Matching Adversarial Model**, for data-efficient image generation with limited training data.

The main idea of SMAM is to simulate the inverse Laplacian pyramid through a neural network. Under an appropriate parameterization, the score function of this neural network is approximately proportional to the negative high-frequency residuals in the forward degradation process. Based on this observation, SMAM introduces a coarse-to-fine generative framework with a strong multi-scale inductive bias.

SMAM mainly consists of two components:

- **Laplacian Pyramid Discriminator (LPD)**  
  The LPD contains a Laplacian pyramid decomposition module, which supports both the forward degradation process and multi-scale discrimination between real and generated images.

- **Inverse Laplacian Pyramid Generator (ILPG)**  
  The ILPG generates the prior of the forward degradation process and predicts the high-frequency residuals required for inverse Laplacian pyramid reconstruction.

Through the inverse Laplacian pyramid reconstruction process, SMAM progressively recovers complete images from generated priors and residual components. The model naturally separates global color and shape information from high-frequency details, which improves generation quality and robustness in limited-data scenarios.

## Abstract

This paper demonstrates that a robust and data-efficient image generation model can be achieved by simulating the inverse Laplacian pyramid through a neural network. The key insight is that under appropriate parameterization, the score function of this neural network is approximately proportional to the negative high-frequency residuals in the forward Degradation process. We refer to this framework as the Score-Matching Adversarial Model (SMAM). The model consists of two main components: the Laplacian Pyramid Discriminator (LPD) and the Inverse Laplacian Pyramid Generator (ILPG). The LPD includes a Laplacian pyramid decomposition module, which enables both the forward degradation process and multi-scale image discrimination for real and generated images. The ILPG, in turn, generates the prior of the forward degradation process as well as the high-frequency residuals needed for the inverse Laplacian pyramid reconstruction. This allows the Laplacian pyramid reconstruction module to recover a complete image from the generated data. SMAM is grounded in a solid theoretical framework and incorporates a strong coarse-to-fine inductive bias. The model’s inherent multi-scale nature and its ability to disentangle global color and shape features contribute to its superior performance in data-limited settings. SMAM significantly outperforms baseline models, and it performs on par with other leading methods in comparative experiments. Extensive experiments prove the effectiveness and superiority of SMAM.

## Environment

The code was developed and tested with the following environment:

```bash
python==3.7
pytorch==1.7.1
cudatoolkit==10.2
