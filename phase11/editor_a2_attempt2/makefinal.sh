#!/bin/bash
# Create final-scaling variants from pristine baseline.
ED=/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor
PY=~/wsp3c-venv/bin/python
cd /home/user/wsp11/sandbox || exit 1
make() { dst=$1; shift; "$PY" "$ED/edit_model.py" model "$dst" "$@" >/dev/null 2>&1 && echo "made $dst"; }
make m_dn15  'scale:mlp.down_proj.weight:1.15'
make m_dn085 'scale:mlp.down_proj.weight:0.85'
make m_edn115 'scale:experts.down_proj:1.15'
make m_opr110 'scale:self_attn.o_proj.weight:1.1'
make m_ls29 'layerscalar:29:0.7'
