-- creating a virtual environment for mac
python -m venv fs_rag
activate - source fs_rag/bin/activate

for windows
first step will remain the same
python -m venv "name"
to activate
source name\Scripts\activate


-- we convert working directory to local or git repo using git init command
-- to track file we write git add filename, git add .
-- git status to check the files that I am tracking

git remote add origin https://github.com/Zameerkhan16/z_rag.git
git branch -M main
git push -u origin main

-- pip install -r requirements.txt