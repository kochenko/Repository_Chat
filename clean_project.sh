#!/bin/bash

# Remover arquivos de cache do Python
find . -name "*.pyc" -exec rm -f {} \;
find . -name "__pycache__" -exec rm -rf {} \;

# Remover logs
find . -name "*.log" -exec rm -f {} \;

# Remover arquivos de backup (~)
find . -name "*~" -exec rm -f {} \;

# Remover arquivos temporários
find . -name "*.tmp" -exec rm -f {} \;
find . -name "*.swp" -exec rm -f {} \;

# Remover arquivos grandes que não são necessários
find . -type f -size +100M -exec rm -f {} \;

# Remover diretórios desnecessários
rm -rf .git
rm -rf node_modules
rm -rf .idea
rm -rf .vscode
rm -rf .github

# Remover arquivos específicos que você sabe que não são necessários
rm -f README.md
rm -f LICENSE
rm -f requirements.txt

# Remover arquivos e diretórios relacionados ao Docker (se não forem necessários)
rm -rf docker
rm -f Dockerfile
rm -f docker-compose.yml

# Remover arquivos e diretórios relacionados a testes (se não forem necessários)
rm -rf tests

# Adicione outros arquivos ou diretórios específicos que você sabe que podem ser removidos

echo "Limpeza concluída!"

