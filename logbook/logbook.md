# Logbook

## Meeting on 30th, May

### Project Background and Objectives
- **Origin**: Evolved from signal analysis techniques in the oil and gas industry, particularly seismic data for feature identification. The goal is to transfer this methodology to medical imaging.
- **Objectives**:
  - Directly extract meaningful features from MRI raw data (k-space) without traditional image reconstruction.
  - Initially validate using **phantoms**, then extend to **clinical datasets**.
  - Expected outcomes:
    - Develop a novel MRI feature extraction method.
    - Publish academic papers.
    - Explore commercial applications (e.g., standalone tools or platforms).


### Technical Approach
- **Data Processing**:
  - Focus: Raw k-space MRI data.
  - Methodology: Signal decomposition and feature detection, different from traditional image-based AI analysis.
- **Experiment Workflow**:
  1. Validate feasibility using phantom experiments (simulated human tissue).
  2. Transition to clinical real-world datasets (subject to ethical approvals).
  3. Primary application: Cardiac MRI analysis.


### Upcoming Tasks and Requirements
- **IRP Project Plan**:
  - Deadline: Within two weeks (by June 14).
  - Grade weight: The IRP accounts for 30% of the total course grade, and the project plan contributes around 5% of the IRP grade.
  - Writing Tips:
    - Week 1: Read extensively on related literature and identify research gaps.
    - Week 2: Draft a clear and feasible project plan based on readings.
    - Note: Plans can evolve during the project; a clear initial structure helps manage changes smoothly.
- **Resources**:
  - Anthony will provide reference materials and previous meeting recordings.
  - A Dropbox will be set up for resource sharing.
- **Site Visit**:
  - Arrange a visit to the Mayo Clinic London MRI facility to understand the operation and data collection process.
- **Communication**:
  - Weekly regular meetings (tentatively Thursday or Friday).
  - A WhatsApp group will be established for real-time communication.


### Other Notes
- **Internal Supervisor**: Assigned by Imperial College, mainly from computational background. Medical expertise is covered by the team.
- **Email Contact**: Maria’s NHS email address will be used for follow-ups.
- **Next Steps**:
  - Anthony will send the project plan draft and reading list.
  - Participants should submit their project plan drafts by the specified deadline for review and grading.



## Meeting on 6th, June
### Project Plan
- should upload to GitHub brfore Friday.
- work out a draft before Wednesday and we could send to Dr. Sommerfeld for feedback.

### Project Process
- Try phantom data first.
- Every method or dataset could be ok. Try to be active, embracing different method.




## Meeting on 13th, June
Online
### Topic:  
Initial brainstorming on project direction and feasibility.  

### Discussion Points:  

- Agreed that the project should explore **novel ways of using MRI k-space data**, not just image-space reconstruction.  

- Raised the question of whether the task should be **segmentation, classification, or both**, and how to define a feasible clinical target.  

- Highlighted the need to understand **what information is preserved in magnitude vs. phase of k-space**, and why that might matter clinically.  

- Noted that **public datasets** (fastMRI, CMRxRecon) may not include sufficient labels for disease detection; alternative options include **synthetic tasks or proxy objectives**.  

- Emphasized the importance of defining **evaluation metrics** early (e.g., Dice, SSIM, classification accuracy), so that project goals are measurable.  

- Agreed to prepare a **literature review summary** before the next meeting to better situate the project in context.  




## Meeting on 20th, June
Offline

### Topic:  
Discuss what data do we need and expected.

### Discussion Points:

- Confirmed that the project will focus on directly using raw k-space data rather than reconstructed images.

- Identified the need for paired k-space and clinical labels, preferably at study-level (e.g. patient-level diagnosis) rather than slice-level annotations.

- Discussed importance of having fully sampled k-space as a baseline, and the option to experiment with undersampled versions later.

- Debated between using public datasets (e.g., fastMRI single-coil, multi-coil) and in-house Cine MRI data, depending on availability of labels and consistency.

- Considered different classification tasks.



## Meeting on 27th, June
### Topic:  
Exploring direct learning from k-space data without full image reconstruction

### Discussion Points:  

- Traditional workflows rely on IFFT to convert k-space to image domain before applying downstream tasks like classification or segmentation.
- Recent studies (e.g., fastMRI baseline) indicate that full image reconstruction is resource-intensive and error-prone in low-SNR or undersampled scenarios.
- Team raised the possibility of bypassing reconstruction using deep learning—early experiments with end-to-end models were discussed.
- Highlighted the success of models like WF-FNO and initial trials of GRU-based sequence processing for k-space slices.
- Identified the challenge: training stability and interpretability of features in raw k-space.

### Suggestions / Action Items:  

- Investigate dimensionality reduction techniques (e.g., PCA, masking) to preprocess k-space before feeding into models.
- Evaluate existing end-to-end k-space models on simplified tasks (e.g., binary classification) before attempting segmentation.
- Start with synthetic or well-curated datasets to validate feasibility of skipping IFFT.
- Review literature on physics-informed or hybrid models that combine partial reconstruction with direct learning.
- Prepare a baseline: use a shallow CNN to classify undersampled k-space (e.g., normal vs. pathology).


## Meeting on 18th, July
AS Comments:  
Image reconstruction from raw k-space is a high-complexity task, and it often needs thousands of well-aligned paired examples.  
However, your primary aim does not need reconstruction. We know AI models can classify, detect, or segment features directly from k-space and we know it works from Imogen.
 

Suggestions:  
Instead of image generation, maybe focus on a simpler detection or classification task (e.g. normal vs abnormal, or structure A vs B).  
Consider an autoencoder or U-Net-style approach to compress k-space and decode it into an interpretable output.



## Meeting on 31th, July

### Visiting Mayo Clinic!

Meeting details:

This enhancement significantly increases the potential for a unified pipeline integrating our respective contributions.  

To clarify the integration strategy moving forward:  

Our current work (Vera and I) focuses on feature extraction, which includes detecting and localizing specific features of interest—e.g., through bounding boxes or spatial coordinates.  

- Imogen’s updated model would then take these identified features and further classify their material composition, distinguishing between fluid, calcium, edema, and potentially other pathological or structural materials.  

- Vera’s model employs self-supervised learning to embed k-space features, which presents a powerful foundation for downstream tasks such as detection or classification.  

- A critical aspect of her approach lies in the design of contrastive pair transformations, including coil dropout and frequency jitter. However, she raised an important point: these transformations might need to be adapted across different datasets or modalities, especially if we aim to integrate models into a joint pipeline.  

Therefore, data source alignment and standardisation of transformation strategies should be established as a first step before integration to ensure consistency and effective feature correspondence.  

This division of labor and modular design makes it feasible to link our pipelines into a comprehensive diagnostic or analysis system. I'm excited about the synergy between our models and the possibilities for more sophisticated and interpretable outputs. Vera expressed enthusiasm for the integration and proposed syncing up with me and Imogen to discuss next steps or even sketch a simple proof of concept. I’m fully on board and would be happy to arrange a time for this collaborative planning session.





## Meeting on 8th, August
Meeting in person.

Meeting details:
### Topic:  
Refining project scope and dataset strategy.  

### Discussion Points:  

- Decided to start with segmentation tasks at study-level.  

- Agreed that fastMRI single-coil data will be the initial dataset, with the option to include Cine MRI later.  

- Planned to implement a baseline ResNet model on reconstructed images for quick benchmarking.  

- Yiding proposed an alternative strategy: convert available image-domain datasets into synthetic k-space, enabling experiments even when raw k-space with labels is scarce.  

- Discussed that this approach could bridge the gap in study-level annotations, allowing initial model development before fully labelled raw data is available.  







## Meeting on 15th, August

### Some discussions about AI applications（with Prof. Barber from UCL）

The discussion covered multiple aspects of large language models (LLMs), their reasoning capabilities, and potential applications. Key points included:  

- Reasoning and Tool Use – LLMs are currently strong in high-level reasoning but less reliable in precise calculation or logical inference. A promising approach is to have LLMs recognize problem types and call specialized tools (e.g., calculators, inference engines) to achieve accurate results.

- Handling Information and Uncertainty – There was exploration of how LLMs process prior knowledge when presented with new evidence, with suggestions to incorporate Bayesian-style confidence assessment.

- Applications in Finance – Participants discussed using LLMs for news sentiment analysis and investment decision-making, with methods ranging from brute-force predictive modeling to manual feature extraction.

- Graph Neural Networks in Investment Prediction – A case study on link prediction between investors and funds highlighted data sparsity issues, negative sampling strategies, and model stability improvements.

- Physics Simulation Models – The limitations of models trained on synthetic data were noted, particularly in discovering new physical laws, though opportunities exist for accelerating simulations.

- Neurobiological-Inspired Architectures – Modular and routing networks were discussed as potential ways to better mirror brain functions and improve adaptive decision-making in AI.

- Future of AI and Human Roles – LLMs may evolve into orchestrators of specialized tools rather than performing all tasks internally, with humans guiding tool selection and application.

- Academic Advice – For MSc/PhD projects, clear problem statements, logical structure, high-quality writing, and evidence of scientific thinking are crucial for strong evaluations, regardless of whether results meet initial expectations.




## Meeting on 22th, August

### Paper Feedback Summary

#### Key Strengths
- **Undersampling results** are the strongest contribution.  
  - Finding: K-space models degrade less under severe under-sampling.  
  - Significance: This is clinically important and should be emphasized more prominently in the paper.

### Points for Improvement

#### Methodology
- **Fusion strategy (K-space + image)**:  
  - Current explanation of feature combination appears too simplistic (just concatenation).  
  - Need to clarify and justify the design or consider exploring more sophisticated fusion approaches.

#### Writing Quality
- **Excessive jargon without explanation**:  
  - Terms like *“SVD-based coil compression”*, *“variable-density Cartesian undersampling”*, and *“Auto-Calibration Signal lines”* require short definitions when first introduced.  

- **Overly complex sentences**:  
  - Example:  
    - Original: *“Each sampling point in K-space contains global spatial frequency information, which jointly determine the formation of the final image.”*  
    - Suggested: *“Each K-space point encodes spatial frequency information that collectively determines the final image structure.”*

#### Clinical Motivation
- The clinical relevance needs to be made stronger.  
  - Clarify **why this research matters for patient care**, not just its technical novelty.  
  - Highlight potential impact on diagnosis, treatment decisions, or workflow efficiency.

---

### Action Items
1. Emphasize **undersampling results** more strongly in abstract, introduction, and discussion.  
2. Provide a clearer and more robust explanation of the **fusion strategy**.  
3. Revise writing:  
   - Add short explanations for technical jargon.  
   - Simplify unnecessarily complex sentences.  
4. Strengthen **clinical motivation** section by connecting technical results to patient care impact.





## Notes on 28th, August

## Key Revisions Needed for the Report

### 1. Abstract
- Clearly state the novel contribution compared to existing works (e.g., AUTOMAP, ETER-net).  
- Avoid sounding like a mere feasibility check—highlight what is genuinely new.  

### 2. Methods – Dataset & Preprocessing
- Provide exact dataset breakdown (how many scans, which subsets, SAX slices, etc.).  
- Justify design choices (e.g., why R=24 undersampling, how much variance retained with 8 virtual coils).  

### 3. Model & Training
- Improve Figure 4 annotation so terms like "SweepGRU" are understandable immediately.  
- Add hyperparameters and hardware details (epochs, batch size, learning rate schedule, random seeds, GPU used) for reproducibility.  

### 4. Results
- Compare against a baseline IFFT reconstruction to establish context.  
- Report variance/error bars across folds instead of only single-point Dice/mIoU values.  
- Emphasize the hybrid K-space + image input as the main finding.  

### 5. Discussion & Clinical Angle
- Reflect on why K-space features struggle in fine segmentation (lack of high-frequency detail, localisation issues).  
- Discuss clinical feasibility (real-time execution, scanner integration).  

### 6. Ethics & Limitations
- Add a short Ethics subsection: de-identification, dataset bias (all healthy volunteers), and clinical safety concerns.  
- Stress that no pathology data was used, so diagnostic claims remain unproven.  







