Ms. Ref. No.: KNOSYS-D-26-03771
Title: PDT: Predictive Domain Transform for Multivariate Time Series Forecasting
Knowledge-Based Systems

Dear Prof. Jian'gang Lu,

Enclosed please find the reviews of the above manuscript. On the basis of the reviewers' appraisals and my own, your manuscript is conditionally accepted pending revision.


Please be aware that, as recommended by the referees, kindly use the English Language Editing service available from Elsevier's Author Services . In your cover letter please provide a point-by-point response to each comment.

When I have read your revised manuscript, I will notify you of the status of your manuscript.

* While submitting revision, please ensure to submit a list of changes or a rebuttal against each point that was being
     raised when you submit the revised manuscript.
* The revised cover letter, response to reviewers, highlights, revised manuscript, credit author statement, author
    agreement and declaration of interest should all be in word format.
* Source file alone can be in word or Latex file and only one final version of manuscript is only required
* The tables, figures and equations should be in editable format in the revised manuscript.
* Figures should be in good quality 300 dpi and not in pdf format
* For LaTex version, kindly ensure to check if supporting .bib files are also included.


PLEASE NOTE: Knowledge-Based Systems would like to enrich online articles by displaying interactive figures that help the reader to visualize and explore your research results. For this purpose, we would like to invite you to upload figures in the MATLAB .FIG file format as supplementary material to our online submission system. Elsevier will generate interactive figures from these files and include them with the online article on SciVerseScienceDirect. If you wish, you can submit .FIG files along with your revised submission.

Please note that this journal offers a new, free service called AudioSlides: brief, webcast-style presentations that are shown next to published articles on ScienceDirect (see also http://www.elsevier.com/audioslides). If your paper is accepted for publication, you will automatically receive an invitation to create an AudioSlides presentation.


Please note that we allow 21 days for the first author revision and 14 days for any additional author revisions that are required. The due date for submitting a revised manuscript files on May 26, 2026.


Research Elements (optional)
This journal encourages you to share research objects - including your raw data, methods, protocols, software, hardware and more – which support your original research article in a Research Elements journal. Research Elements are open access, multidisciplinary, peer-reviewed journals which make the objects associated with your research more discoverable, trustworthy and promote replicability and reproducibility. As open access journals, there may be an Article Publishing Charge if your paper is accepted for publication. Find out more about the Research Elements journals at https://www.elsevier.com/authors/tools-and-resources/research-elements-journals?dgcid=ec_em_research_elements_email.

Yours sincerely,

Hang Yu
Senior Editor
Knowledge-Based Systems



Editor comments:
Revise (Including Language Editing)

Reviewers' comments:



Reviewer #1: This paper proposes a Predictive Domain Transform (PDT) framework for multivariate time series forecasting (MTSF). The core idea is to learn a data-adaptive predictive domain transform (PDTM) that projects time series into a prediction-optimal latent space with strong energy concentration, thereby alleviating the burden on downstream models. Building upon this, the authors design a dual-route architecture, consisting of a Linear-Route for dominant component extrapolation and an Encoder-Route for modeling nonlinear dynamics. In addition, a Masked Channel Dependent (MCD) mechanism is introduced to selectively capture inter-channel dependencies using a lightweight linear-attention formulation. Extensive experiments on multiple benchmark datasets demonstrate competitive performance in both accuracy and efficiency. In general, this is a solid work with clear organization and presentation. I have the following suggestions for the authors:

1) The motivation of PDTM could be further clarified, especially its distinction from existing data-adaptive transforms (e.g., FreDF, TransDF). A more explicit comparison at the conceptual level would improve clarity.

2) The derivation of the predictive domain transform (Section 3.2) is mathematically sound, but some intermediate steps are condensed. Providing additional intuition (e.g., connection to CCA or low-rank regression) would enhance readability for a broader audience.

3) The notation in the methodology section could be improved for consistency, particularly the transitions between matrices (e.g., B, G, W) and covariance terms. A summary table of symbols would be helpful.

4) The choice of hyperparameters, such as the reduced rankr, Top-K components, and mask threshold parameters in MCD, is not sufficiently discussed. Including sensitivity analysis or guidelines for selecting these parameters would strengthen the paper.

5) Although the Masked Channel Dependent (MCD) strategy is well-motivated, the interpretability of the learned mask is not explored. Visualizing or analyzing the learned channel relationships could provide additional insights.



Reviewer #3: 1. It is recommended that the experimental section include quantitative tests of robustness and boundary conditions: In the visual analysis in Section 4.3, the paper mentions that the model exhibits strong resilience to occasional noise events and that the MCD module is designed to 'suppress noise in irrelevant channels'. However, the experimental section lacks a quantitative analysis of noise resistance. It is recommended that the authors employ more rigorous anomaly injection tests to verify this.
2. It is recommended to include sensitivity analyses for key hyperparameters:
The paper proposes the extraction of Top-K components using Linear-Route in Section 3.3.3, and introduces thresholds and steepness for generating channel masks in Section 3.3.4. Ablation experiments and sensitivity analyses for these key hyperparameters should be included.
3. The experiments utilised only two scale-dependent metrics: MSE and MAE. When applying the model across datasets from different domains, there is a lack of relative error evaluation; it is recommended to include metrics such as RMSE.
4. The baselines selected for the experiments include mainstream Transformer and Linear models, but overlook some recent advanced graph neural network time series forecasting models specifically designed to explicitly model the topological structure of multivariate variables.
5. The paper mentions that PDTM can concentrate 'predictive information' into the Top-K features; however, this is an empirical observation, and there is no proof as to why eigenvalue decay necessarily holds, or whether it depends on the data distribution.
6. It is recommended that the authors provide a link to an anonymised code repository in the revised manuscript, clearly indicating the parameter settings, to verify the authenticity of the MACs and Memory computation results.



*********************************************
For further assistance, please visit our customer support site at http://help.elsevier.com/app/answers/list/p/7923. Here you can search for solutions on a range of topics, find answers to frequently asked questions and learn more about EM via interactive tutorials. You will also find our 24/7 support contact details should you need any further assistance from one of our customer support representatives.

At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

#AU_KNOSYS#

To ensure this email reaches the intended recipient, please do not delete the above code