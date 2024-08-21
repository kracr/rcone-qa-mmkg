# Scene Graph Generation Module

## Environment preparation
+ Set environment as - https://github.com/liuhengyue/fcsgg 
+ download ```VGG``` data as described in their readme file

## Data download
Download checkpoint folder, datasets folder and output folder from the following 3 links

+ https://drive.google.com/file/d/1BB7EJdo7On79vv49gQtvEUomHOaB20iS/view?usp=sharing

+ https://drive.google.com/file/d/1gzAO1I_gRdlx0PVxsLz8CRVFoAg4DYI_/view?usp=sharing

+ https://drive.google.com/file/d/1PfOYZIv_rQyTglOfjEc8mf8ek0jCSmgW/view?usp=sharing

unzip all the downloaded folders

## Training
run 
    
    sh script_train.sh 
It contains both the models for query generation and training of RConE

### SGG generation
run 
    
    sh script.sh 
It generate scene graph for ```fb15k``` dataset, change the config of the two models (```32``` and ```48```), in custom_predict.py (described in script), to choose generated scene graph (either for query generation or training)
+ other datasets have same image set as fb15k, as dicussed in *Liu, Ye, et al. MMKG: multi-modal knowledge graphs. The Semantic Web: 16th International Conference, ESWC 2019, Portorož, Slovenia, June 2–6, 2019, Proceedings 16. Springer International Publishing, 2019.*
+ keep the result scene graph in ```../models/results``` folder, as currently present 
