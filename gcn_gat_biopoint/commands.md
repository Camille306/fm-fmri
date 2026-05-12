For gat: 
python main.py --train --test --model_type gat     -dt ./results/gat_real -n gat_real     -b 8 --top_k_edges 50
python main.py --train --test --model_type gat --use_synthetic     --synthetic_dir ~/rest_to_task/stagin_biopoint/synthetic_biopoint     -dt ./results/gat_syn_tuned -n gat_syn_tuned     -b 16 --top_k_edges 50     --quality_frac 0.30     --hidden_dim 64 --dropout 0.5     --lr 1e-4 --weight_decay 0.01     --scheduler_step 20 --scheduler_gamma 0.5     --num_epochs 100     --window_num 6 --window_size 60

For gcn: 
python main.py --train --test --model_type gcn --use_synthetic     --synthetic_dir ~/rest_to_task/stagin_biopoint/synthetic_biopoint     -dt ./results/gcn_syn_tuned -n gcn_syn_tuned     -b 16 --top_k_edges 50     --quality_frac 0.15     --hidden_dim 64 --dropout 0.5     --lr 1e-4 --weight_decay 0.01     --scheduler_step 20 --scheduler_gamma 0.5     --num_epochs 50     --window_num 4 --window_size 60
python main.py --train --test --model_type gcn     -dt ./results/gcn_real -n gcn_real     -b 4 --top_k_edges 50