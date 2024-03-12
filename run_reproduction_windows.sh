#####################################################################################
# BASH SCRIPT FOR REPRODUCING THE RESULTS IN CROSSWALK: FAIRNESS-ENHANCED NODE      #
# REPRESENTSTION LEARNING                                                           #
#                                                                                   #
#                                                                                   #
#####################################################################################


### 1. MAKE EMBEDDINGS ###

#SETTINGS
declare MAKE_EMBEDDINGS=true
declare RUN_INFMAX=true
declare RUN_LINKPRED=true
declare RUN_NODECLASS=true

declare PPATH=                                              # local path to python.exe
declare WALK_LENGTH=5                                       # random walk length for CrossWalk
declare P=4.0                                               # exponent for CrossWalk
declare SEED=0                                              # seed for embeddings

# Initialise used methods

declare DATASETS=("rice" "twitter" "synth2" "synth3")
declare METHODS=("deepwalk" "fairwalk" "crosswalk")
declare ITERS=("1" "2" "3" "4" "5") # number of iterations for each used method


# MAKE EMBEDDINGS
if [ ${MAKE_EMBEDDINGS} = true ]; then 
    echo "datasets: ${DATASETS[@]}"
    for dataset in ${DATASETS[@]}; do
        echo Analyzing ${dataset}
        echo "======================================================================================="
        declare GRAPH_DIR="data/${dataset}"
        declare ROOTNAME="${dataset}_graph"
        if [ ${dataset} = "rice" ] || [ ${dataset} = "twitter" ]; then
            declare TESTLINKS=0.1
            declare ALPHA=0.5
        elif [ ${dataset} = "synth2" ] || [ ${dataset} = "synth3" ]; then
            declare TESTLINKS=0.
            declare ALPHA=0.7
        fi

        echo "embedding settings:"
        echo "--------------------------------------------------------------------------------------"
        echo "Used methods: ${METHODS[@]}"
        echo "CrossWalk parameters: walklength = ${WALK_LENGTH}, alpha = ${ALPHA}, p = ${P}"
        echo "number of iterations: ${ITERS[-1]}"
        echo "--------------------------------------------------------------------------------------"
        echo "creating embeddings..."
        echo "=========="
        for method in ${METHODS[@]}; do
            if [ ${method} = "deepwalk" ]; then 
                declare weight_method="unweighted"
            
            elif [ ${method} = "fairwalk" ]; then 
                declare weight_method="fairwalk"
            
            elif [ ${method} = "crosswalk" ]; then 
                declare weight_method="random_walk_${WALK_LENGTH}_bndry_${ALPHA}_exp_${P}"
            

            fi

            echo "${method} , ${weight_method}"
            for iter in ${ITERS[@]}; do
                echo "${iter}"
                echo "----------"
                ${PPATH}/python.exe make_embeddings --format edgelist\
                                --input ${GRAPH_DIR}/${ROOTNAME}.links\
                                --max-memory-data-size 50000000\
                                --number-walks 30\
                                --representation-size 128\
                                --walk-length 40\
                                --window-size 10\
                                --workers 30\
                                --output ${GRAPH_DIR}/embeddings/${method}/${ROOTNAME}_${method}_${iter}.embeddings\
                                --weighted ${weight_method}\
                                --sensitive-attr-file ${GRAPH_DIR}/${ROOTNAME}.attr\
                                --test-links ${TESTLINKS}\
                                --train-links-file ${GRAPH_DIR}/trainlinks/${method}/${ROOTNAME}_${method}_${iter}.trainlinks\
                                --test-links-file ${GRAPH_DIR}/testlinks/${method}/${ROOTNAME}_${method}_${iter}.testlinks\
                                --seed ${SEED}

                if [ ${dataset} = "rice" ]; then
                    echo "make Node2Vec embedding..."
                    ${PPATH}/python.exe influence_maximization --input_file ${GRAPH_DIR}/${ROOTNAME}.links\
                                    --sens_attr_file ${GRAPH_DIR}/${ROOTNAME}.attr\
                                    --method ${method}\
                                    --weighted ${weight_method}\
                                    --nx_graph_file ${GRAPH_DIR}/${ROOTNAME}_nx_${method}.links\
                                    --output_file ${GRAPH_DIR}/N2Vembeddings/${method}/${ROOTNAME}_${method}_${iter}.n2vembeddings
                fi
            done
            echo "----------"
        done
    done
    echo "=========="
    echo "embeddings created; saved in ${GRAPH_DIR}/embeddings"
    echo "======================================================================================="
fi
## 2. RUN INFLUENCE MAXIMIZATION ALGORITHM ###


# RUN INF MAX
if [ ${RUN_INFMAX} = true ]; then
    echo "running influence maximization..."
    for dataset in ${DATASETS[@]}; do
        echo Analyzing ${dataset}
        echo "======================================================================================="
        declare GRAPH_DIR="data/${dataset}"
        declare ROOTNAME="${dataset}_graph"
        if [ ${dataset} = "rice" ] || [ ${dataset} = "twitter" ]; then
            declare ACT_WEIGHT=0.01
        elif [ ${dataset} = "synth2" ] || [ ${dataset} = "synth3" ]; then
            declare ACT_WEIGHT=0.03
        fi

        for method in ${METHODS[@]}; do
            echo "${method}"
            echo "----------"
            for iter in ${ITERS[@]}; do
                echo "${iter}..."
                # $PPATH/python.exe influence_maximization --graph_file ${GRAPH_DIR}/${ROOTNAME}\
                #                 --act_weight ${ACT_WEIGHT}\
                #                 --input ${GRAPH_DIR}/embeddings/${method}/${ROOTNAME}_${method}_${iter}.embeddings\
                #                 --output ${GRAPH_DIR}/infmaxresults/${method}/${ROOTNAME}_${method}_${iter}                  
            
                if [ ${dataset} = "rice" ]; then
                    echo "run infmax with Node2Vec..."
                    $PPATH/python.exe influence_maximization --graph_file ${GRAPH_DIR}/${ROOTNAME}\
                                --act_weight ${ACT_WEIGHT}\
                                --input ${GRAPH_DIR}/N2Vembeddings/${method}/${ROOTNAME}_${method}_${iter}.n2vembeddings\
                                --output ${GRAPH_DIR}/N2Vinfmaxresults/${method}/${ROOTNAME}_${method}_${iter}_n2v
                fi
            done
            echo "----------"
        done
    done
    echo "=========="
    echo "results obtained; saved in ${GRAPH_DIR}/infmaxresults"
    echo "======================================================================================="
fi

# ## 3. RUN THE LINK PREDICTION ALGORITHM ###

#RUN LINK PRED
if [ ${RUN_LINKPRED} = true ]; then
    echo "running link prediction ..."

    for dataset in ${DATASETS[@]}; do
        if [ ${dataset} = "rice" ] || [ ${dataset} = "twitter" ]; then
            echo Analyzing ${dataset}
            echo "======================================================================================="
            declare GRAPH_DIR="data/${dataset}"
            declare ROOTNAME="${dataset}_graph"
            for method in ${METHODS[@]}; do
                echo "${method}"
                echo "----------"
                for iter in ${ITERS[@]}; do
                    echo "${iter}..."
                    $PPATH/python.exe link_prediction --num_iters ${ITERS[-1]}\
                                    --emb_file ${GRAPH_DIR}/embeddings/${method}/${ROOTNAME}_${method}_${iter}.embeddings\
                                    --sens_attr_file ${GRAPH_DIR}/${ROOTNAME}.attr\
                                    --train_links_file ${GRAPH_DIR}/trainlinks/${method}/${ROOTNAME}_${method}_${iter}.trainlinks\
                                    --test_links_file ${GRAPH_DIR}/testlinks/${method}/${ROOTNAME}_${method}_${iter}.testlinks\
                                    --output_file ${GRAPH_DIR}/lpresults/${method}/${ROOTNAME}_${method}_${iter}_lpresults.txt               
                done
                echo "----------"
            done
        fi
    done
    echo "=========="
    echo "results obtained; saved in ${GRAPH_DIR}/lpresults"
    echo "======================================================================================="
fi


## 4. RUN THE NODE CLASSIFICATION ALGORITHM ###

#RUN NODE CLASS 
if [ ${RUN_NODECLASS} = true ]; then
    echo "running node classification ..."

    for dataset in ${DATASETS[@]}; do
        if [ ${dataset} = "rice" ] ; then
            echo Analyzing ${dataset}
            echo "======================================================================================="
            declare GRAPH_DIR="data/${dataset}"
            declare ROOTNAME="${dataset}_graph"
            for method in ${METHODS[@]}; do
                echo "${method}"
                echo "----------"
                for iter in ${ITERS[@]}; do
                    echo "${iter}..."
                    $PPATH/python.exe node_classification --emb_file ${GRAPH_DIR}/embeddings/${method}/${ROOTNAME}_${method}_${iter}.embeddings\
                                    --label_file ${GRAPH_DIR}/${ROOTNAME}_raw.attr\
                                    --sens_attr_file ${GRAPH_DIR}/${ROOTNAME}_sensitive_attr.txt\
                                    --method ${method}\
                                    --out_file ${GRAPH_DIR}/ncresults/${method}/${ROOTNAME}_${method}_${iter}_ncresults.txt              
                done
                echo "----------"
            done
        fi
    done
    echo "=========="
    echo "results obtained; saved in ${GRAPH_DIR}/ncresults"
    echo "======================================================================================="
fi

echo "All done"