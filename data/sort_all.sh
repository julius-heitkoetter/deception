#!/bin/bash
#---------------------------------------------------------------------------------------------------
# Automatically sort all qaeve datasets into fixed deceiver and fixed supervisor folders
#---------------------------------------------------------------------------------------------------

# Usage: 
#    copy this into the directory where the files are stored
#    `source sort_all.sh`
#    all files will be copied into directories in the folder where this is run

# NOTE: Sometime you may get an error resembling this:
#       cp: cannot stat ‘*_qaeve_gpt-3.5-turbo_*’: No such file or directory
# This is ok, it just means there aren't any of those files there

# Make directories
mkdir llama_7b_deceiver
mkdir llama_13b_deceiver
mkdir llama_70b_deceiver
mkdir gpt_35_deceiver
mkdir llama_7b_supervisor
mkdir llama_13b_supervisor
mkdir llama_70b_supervisor
mkdir gpt_35_supervisor

# Copy over files
cp *_qaeve_Llama-2-7b-chat-hf_* llama_7b_deceiver/.
cp *_qaeve_Llama-2-13b-chat-hf_* llama_13b_deceiver/.
cp *_qaeve_Llama-2-70b-chat-hf_* llama_70b_deceiver/.
cp *_qaeve_gpt-3.5-turbo_* gpt_35_deceiver/.

cp *_Llama-2-7b-chat-hf_202* llama_7b_supervisor/.
cp *_Llama-2-13b-chat-hf_202* llama_13b_supervisor/.
cp *_Llama-2-70b-chat-hf_202* llama_70b_supervisor/.
cp *_gpt-3.5-turbo_202* gpt_35_supervisor/.