#optimum-cli export openvino --trust-remote-code --model  /root/autodl-fs/qwen2_7B_v2_total_plus/merge --task text-generation-with-past --weight-format int4   qwen2_7B_v2_total_plus_ov
#optimum-cli export openvino --trust-remote-code --model  /root/autodl-fs/qwen2_1.5b_v3_full_plus/merge --task text-generation-with-past --weight-format int4  qwen2_1.5b_v3_full_plus_ov
#optimum-cli export openvino --trust-remote-code --model  /root/autodl-fs/qwen2_1.5B_v2_total_plus/merge --task text-generation-with-past --weight-format int4   qwen2_1.5B_v2_total_plus_ov
#optimum-cli export openvino --trust-remote-code --model  /root/autodl-fs/qwen2_1.5B_v1_total_plus/merge --task text-generation-with-past --weight-format int4   qwen2_1.5B_v1_total_plus_ov


tar czvf qwen2_7B_v2_total_plus_ov.tar.gz qwen2_7B_v2_total_plus_ov
tar czvf qwen2_1.5b_v3_full_plus_ov.tar.gz qwen2_1.5b_v3_full_plus_ov
tar czvf qwen2_1.5B_v2_total_plus_ov.tar.gz qwen2_1.5B_v2_total_plus_ov
tar czvf qwen2_1.5B_v1_total_plus_ov.tar.gz qwen2_1.5B_v1_total_plus_ov


tar zxvf qwen2_7B_v2_total_plus_ov.tar.gz
tar zxvf qwen2_1.5b_v3_full_plus_ov.tar.gz
tar zxvf qwen2_1.5B_v2_total_plus_ov.tar.gz
tar zxvf qwen2_1.5B_v1_total_plus_ov.tar.gz




