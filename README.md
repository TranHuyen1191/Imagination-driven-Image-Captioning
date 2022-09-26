# Imagination-driven-Image-Captioning

## Installation 
Creating an environment from environment.yml:
```Console
conda env create -f environment.yml
```

## Create dataset 
 __Run the scripts in CreateDB_Script folder to create IdC-I and -II and pre-process ArtEmis dataset. These datasets will be used to train and test CScorer and image captioning model in the following steps.__

   1. mv CreateDB_Script/* .
   2. Run the [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/CreateDB_Script/CreateDB_1IdCI.ipynb) to extract imagination-driven captions from ArtEmis dataset to create IdC-I.
   3. Run the [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/CreateDB_Script/CreateDB_2IdCII_TypeII.ipynb) and [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/CreateDB_Script/CreateDB_2IdCII_TypeII_AddText.ipynb) to create fake captions of Type-II. 
   4. Run the [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/CreateDB_Script/CreateDB_3IdCII_3type.ipynb) to create fake captions of Type-I and Type-III.
   5. Run the [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/CreateDB_Script/CreateDB_4ArtEmisFromIdC.ipynb) to pre-process and split ArtEmis dataset into training, validation, and test sets. 
   6. Run the [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/CreateDB_Script/CreateDB_5ArtEmis_CreateGT_LCIdC.ipynb) to split the test set into ArtEmis-LC-TS and ArtEmis-IdC-I-TC. 

Datasets can be downloaded at [dataset-link](https://drive.google.com/file/d/1ntsERIJ0gri6om84-cCrpgHe0o9M7SU_/view?usp=sharing), [image-224px-link](), and [image-384px-link]().
Note: Image folders (i.e., CLIP_224 and CLIP_384) should be saved to Dataset/ArtEmis/OriginalArtEmis/Images

## Training 
### CScorer
  * __Train__: Run Metric_adapterCLIP_1FineTune_3MLP.py to fine-tune CLIP.__
     ```Console
    mv CreateDB_Script/Metric_adapterCLIP* .
    python Metric_adapterCLIP_1FineTune_3MLP.py
    ```
  * __Evaluation__: Run the [notebook](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Metric_adapterCLIP_2Eval_3MLP.ipynb) to evaluate the accuracy of CScorer.__

### Image captioning models: 1GEN and 2GEN
  * __Train__: Run Model_ArtEmis_Ours_1GEN2GEN_1Train.py to train the model.__
    - 1GEN: modelname = 'CLIPViTB16_1Gen' 
    - 2GEN: modelname = 'CLIPViTB16_woSG' 
     ```Console
    mv CreateDB_Script/Model* .
    python Model_ArtEmis_Ours_1GEN2GEN_1Train.py
    ```
  * __Generate captions__: Run the notebooks: [1GEN](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Model_ArtEmis_Ours_1GEN_2GenCapt.ipynb) and [2GEN](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Model_ArtEmis_Ours_2GEN_2GenCapt.ipynb) to generate captions for test images from the trained models.

  * __Evaluation__: Run the notebooks: [1GEN](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Model_ArtEmis_Ours_CLIPViTB161Gen_3Eval.ipynb) and [2GEN](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Model_ArtEmis_Ours_CLIPViTB162Gen_3Eval.ipynb) to evaluate the models.
  
  * __Show some examples__: Run the notebooks: [1GEN](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Model_ArtEmis_Ours_CLIPViTB161Gen_Example.ipynb) and [2GEN](https://github.com/TranHuyen1191/Imagination-driven-Image-Captioning/blob/main/ModelEvaluation_Script/Model_ArtEmis_Ours_CLIPViTB162Gen_Example.ipynb) to show some captions generated by the models.
  
### Checkpoints
* Checkpoints of CScorer, 1GEN, and 2GEN can be downloaded at [link](https://drive.google.com/file/d/1RnTwCt6JDt9zHAgVRgXPCy4odnZS_gjt/view?usp=sharing)
    - CScorer: output\adapterCLIP_3MLP\RN50x16_F1\checkpoints\best_model.pt
    - 1GEN: output\Ours_ArtEmis\CLIPViTB16_1Gen\checkpoints\best_model.pt
    - 2GEN: output\Ours_ArtEmis\CLIPViTB16_woSG\checkpoints\best_model.pt

## Authors

* **Huyen Tran** - *RIKEN Center for AIP, Japan*
* **Takayuki Okatani** - *Graduate School of Information Sciences, Tohoku University, Japan*

## Acknowledgments
This code is implemented based on the implementation of "ArtEmis: Affective Language for Visual Art" (https://github.com/optas/artemis) and "CLIP" (https://github.com/openai/CLIP)

If you use this dataset and this source code, please consider to cite their papers and our ACCV paper: 

* Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International Conference on Machine Learning. PMLR, 2021.
* Achlioptas, Panos, et al. "Artemis: Affective language for visual art." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
* Huyen T. T. Tran, T. Okatani. "Bright as the Sun: In-depth Analysis of Imagination-driven Image Captioning." The 16th Asian Conference on Computer Vision (ACCV). 2022. Macau SAR, China.


 
