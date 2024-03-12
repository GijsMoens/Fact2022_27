#####################################################################################
# BASH SCRIPT TO CREATE SYNTHETIC GRAPHS, EMBEDDINGS AND RUN INFLUENCE MAXIMIZATION #
#                                                                                   #
#                                                                                   #
#####################################################################################

### 1. MAKE GRAPHS ###

# SETTINGS
declare MAKE_GRAPHS=true
declare MAKE_EMBEDDINGS=true
declare RUN_INFMAX=true

declare PPATH=                                                      # local path to python.exe
declare SEED=42                                                     # seed for random link initialisation of graph
declare NUM_NODES=1500                                              # number of nodes used in graph
declare NUM_GROUPS=10                                               # number of groups in graph
declare P_GROUPS="400,150,150,150,150,100,100,100,100,100"          # nodes per group
declare PHOM_MIN=0.025                                              # minimimum value of intragroup link probability
declare PHOM_MAX=0.05                                               # maximum of intragroup link probabilty
declare PHET_MIN=0.0015                                             # minimum of intergroup link probability
declare PHET_MAX=0.003                                              # maximum of intergroup link probability
declare GRAPH_DIR="data/synthx${NUM_GROUPS}_${SEED}/${P_GROUPS}"    # path to directory of .links and .attr files

# MAKE GRAPH
if [ ${MAKE_GRAPHS} = true ]; then
    echo "graph settings:"
    echo "---------------------------------------------------------------------------------------"
    echo "seed: ${SEED}"
    echo "groups: ${NUM_GROUPS} (${P_GROUPS})"
    echo "probability ranges: Phom: [${PHOM_MIN} , ${PHOM_MAX}] ; Phet [${PHET_MIN} , ${PHET_MAX}] "
    echo "---------------------------------------------------------------------------------------"
    echo "creating graph..."
    $PPATH/python.exe synthesize_graph_xgroups.py --seed ${SEED}\
                    --nodes ${NUM_NODES}\
                    --num_colors ${NUM_GROUPS}\
                    --Pcolors ${P_GROUPS}\
                    --Phom_min ${PHOM_MIN}\
                    --Phom_max ${PHOM_MAX}\
                    --Phet_min ${PHET_MIN}\
                    --Phet_max ${PHET_MAX}\
                    --output_dir ${GRAPH_DIR}

    echo "graph made; saved in ${GRAPH_DIR}"
    echo "======================================================================================="
fi

### 2. MAKE EMBEDDINGS ###

#SETTINGS
declare ROOTNAME="synth${NUM_NODES}_Phom${PHOM_MIN}-${PHOM_MAX}_Phet${PHET_MIN}-${PHET_MAX}" #core of graph files

declare WALK_LENGTH=8           # random walk length for CrossWalk and MoensWalk
declare ALPHA=0.7               # alpha parameter for CrossWalk and MoensWalk
declare P=4.0                   # exponent for CrossWalk and MoensWalk
declare NORMS=("1" "2" "3" "4") # p-norms for MoensWalk

# Initialise used methods
declare METHODS=("deepwalk" "fairwalk" "crosswalk")

for norm in ${NORMS[@]}; do
    METHODS+=("moenswalk_${norm}")
done


declare ITERS=("1" "2" "3" "4" "5") # number of iterations for each used method

# MAKE EMBEDDINGS
if [ ${MAKE_EMBEDDINGS} = true ]; then
    echo "embedding settings:"
    echo "--------------------------------------------------------------------------------------"
    echo "Used methods: ${METHODS[@]}"
    echo "Cross-/MoensWalk parameters: walklength = ${WALK_LENGTH}, alpha = ${ALPHA}, p = ${P}"
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
            
        elif [[ ${method} = moenswalk* ]]; then
            declare cur_norm=${method##*_}
            declare weight_method="random_walk_${WALK_LENGTH}_bndry_${ALPHA}_exp_${P}_extension_${cur_norm}"
        fi

        echo "${method} , ${weight_method}"
        for iter in ${ITERS[@]}; do
            echo "${iter}"
            echo "----------"
            $PPATH/python.exe make_embeddings --format edgelist\
                            --input ${GRAPH_DIR}/${ROOTNAME}.links\
                            --max-memory-data-size 50000000\
                            --number-walks 30\
                            --representation-size 128\
                            --walk-length 40\
                            --window-size 10\
                            --workers 30\
                            --output ${GRAPH_DIR}/embeddings/${method}/${ROOTNAME}_${method}_${iter}.embeddings\
                            --weighted ${weight_method}\
                            --sensitive-attr-file ${GRAPH_DIR}/${ROOTNAME}.attr
        done
        echo "----------"
    done
    echo "=========="
    echo "embeddings created; saved in ${GRAPH_DIR}/embeddings"
    echo "======================================================================================="
fi

### 3. RUN INFLUENCE MAXIMIZATION ALGORITHM ###

# SETTINGS
declare ACT_WEIGHT=0.03

# RUN INF MAX
if [ ${RUN_INFMAX} = true ]; then
    echo "running influence maximization ..."
    for method in ${METHODS[@]}; do
        echo "___${method}___"
        for iter in ${ITERS[@]}; do
            echo "${iter}..."
            $PPATH/python.exe influence_maximization --graph_file ${GRAPH_DIR}/${ROOTNAME}\
                            --act_weight ${ACT_WEIGHT}\
                            --input ${GRAPH_DIR}/embeddings/${method}/${ROOTNAME}_${method}_${iter}.embeddings\
                            --output ${GRAPH_DIR}/infmaxresults/${method}/${ROOTNAME}_${method}_${iter}                  
        done
    done

    echo "results obtained; saved in ${GRAPH_DIR}/infmaxresults"
    echo "======================================================================================="
fi
echo "All done"

