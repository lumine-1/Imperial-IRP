# Deep Learning in MRI K-space: A Novel Approach to Clinical Decision Support by Directly Learning Raw Signals

This repository provides an end-to-end framework for **cine cardiac MRI segmentation** directly from k-space data.  
It supports multiple models and training pipelines, including:

- **K-space only segmentation**
- **Image only segmentation**
- **Hybrid K-space + Image segmentation**
- Both **fully-sampled** and **undersampled** MRI data

The project is modularized into `models/`, `process/`, and `utils/` packages for clarity, and comes with detailed docstrings and auto-generated documentation.

---
## First introduces the required materials
---

#### Final Report
The final paper of this project.  
Click: [yh924-final-report.pdf](deliverables/yh924-final-report.pdf)

#### Logbook
The record of meetings with supervisor during the project.  
Click: [logbook.md](logbook/logbook.md)

#### LaTeX-related Document: 
In addition to the PDFs, the source files, such as the .tex and figure files for LaTeX are committed to the repository.    
Click: [tex_document](code_and_materials/tex_document)


#### Refinement Record: 
It recorded my refinements after the first draft.  
Click: [Report_refine.txt](code_and_materials/Report_refine.txt)

#### References: 
Record AI usage.  
Click: [reference.md](code_and_materials/reference.md)


#### I have also completed  the user manual and the documentation for the code, which will be introduced in the code section below.

---
## Next introduces the content of the project
---

## 📂 Dataset

This project is built for the **CMRxRecon2023 dataset**.  
The **FastMRI** dataset is also employed in this project, only used in fastmri.py.

Paths to the raw data should be updated in **`settings.py`**:

- `PATH_K`: Path to raw cine MRI k-space validation set  
- `PATH_FULL_OUT`: Output directory for reconstructed fully-sampled images  
- `PATH_UNDER_OUT`: Output directory for reconstructed undersampled images  

---

## ⚙️ Environment Setup

Follow the installation guide in **[prepare_environment.md](code_and_materials/prepare_environment.md)**.

- Try the **minimal environment** (`min_enviroment.yml`) first (recommended).  
- If installation fails, use the **full environment** (`full_enviroment.yml`).

---

## ⚙️ Running the Code

For usage examples, training, evaluation, and visualization, see:  

**[KNet_usage_guide.ipynb](code_and_materials/KNet_usage_guide.ipynb)**

This notebook provides a hands-on demo of the full pipeline:
1. Dataset preparation  
2. Model training (fully sampled / undersampled)  
3. Evaluation  
4. Visualization  

As the model weight file used in this project is relatively small, I have uploaded it to this repository at:[model_weights](code_and_materials/KNet/runs/checkpoints)   
These weights can be directly used for reproduction of the results.

---

## 📖 Documentation

All core modules (`models/`, `process/`, `utils/`) include detailed docstrings.  
We also built Sphinx-generated documentation for exploring the codebase in depth.

**Check the document for code details here : [Sphinx_document.pdf](code_and_materials/Sphinx_document.pdf)!**

If you want to generate html document:

Run from `docs/`:

```bash
make html
```

Output HTML docs will be in `docs/build/html/`.

---

## 📑 Project Structure

```
KNet/
├── models/          # Model definitions (K-space, Image, Hybrid)  
├── process/         # Training, evaluation, visualization, data preparation  
├── utils/           # Helper functions (seed, recon, visualization utils)  
├── docs/            # Sphinx documentation  
├── runs/            # Training outputs (checkpoints, logs)  
├── prepared/        # Preprocessed datasets  
├── main.py          # Entry point for running full pipeline  
├── settings.py      # Centralized configuration file  

min_enviroment.yml   
full_enviroment.yml  
KNet_usage_guide.ipynb
```

---

## Highlights

- Unified pipeline for cine MRI segmentation experiments  
- Modular design for easy extension and reproducibility  
- Comprehensive docstrings + generated documentation  
- Example notebook for quick start  


---

## Acknowledgments

We gratefully acknowledge the providers of the datasets used in this project.  

- F. Knoll et al., *"fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning"*, **Radiol. Artif. Intell.**, vol. 2, no. 1, p. e190007, Jan. 2020.  
  doi: [10.1148/ryai.2020190007](https://doi.org/10.1148/ryai.2020190007)

- J. Zbontar et al., *"fastMRI: An Open Dataset and Benchmarks for Accelerated MRI"*, **arXiv preprint**, 2018.  
  doi: [10.48550/arXiv.1811.08839](https://doi.org/10.48550/arXiv.1811.08839)

- C. Wang et al., *"Recommendation for Cardiac Magnetic Resonance Imaging-Based Phenotypic Study: Imaging Part"*, **Phenomics**, vol. 1, no. 4, pp. 151–170, Aug. 2021.  
  doi: [10.1007/s43657-021-00018-x](https://doi.org/10.1007/s43657-021-00018-x)

As required, if you use these dataset in your research, please also follow the rules and cite works above.

---

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project, provided that the original license and copyright notice are included.

