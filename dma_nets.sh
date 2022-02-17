# bash dma_nets.sh 0 patent_1t1n_main
# bash dma_nets.sh 0 patent_1t1n_sub
# bash dma_nets.sh 0 paper

# =============================================================================
# ====================== Input data and configurations ========================
# =============================================================================
patent="dataset/patentsview"
paper="dataset/mag"
if [[ $2 == 'patent_1t1n_main' ]]; then
    data="$patent/config_1t1n_main.ini"
elif [[ $2 == 'patent_1t1n_sub' ]]; then
    data="$patent/config_1t1n_sub.ini"
elif [[ $2 == 'paper' ]]; then
    data="$paper/config.ini" 
fi

# =============================================================================
# ============================= Run experiments ===============================
# =============================================================================
for ratio in 0.8 0.5 0.3 0.1
do
    echo $data
    python main.py --gpu --use-cuda $1 --ob-ratio $ratio --config $data \
        --t-emb-size 32 --e-emb-size 32 --dq 64 --dk 64 --dv 64 \
        --hidden-size 256 --n-heads 4 --m1 4 --m2 2 --dropout 0.3 \
        --lr 0.0001 --epochs 30 --batch-size 16 --weight-decay 0.0002
done
