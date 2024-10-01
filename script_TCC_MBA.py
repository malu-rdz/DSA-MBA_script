# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:48:37 2024

@author: Maria Luisa de Barros Rodrigues

Script elaborado como parte do TCC do MBA em Data Science and Analytics - turma 231

"""


#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as stats
from scikit_posthocs import posthoc_dunn


#%%

# Preparo do banco de dados - especificação de arquivos

# Nomes do arquivos de entrada
allpop_previous = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\BD_allpop_500MH.txt"
nampop_previous = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\BD_nampop_500MH.txt"

# Nomes dos arquivos de saída - banco de dados
allpop_bd = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\BD_allpop_500MH.xlsx"
nampop_bd = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\BD_nampop_500MH.xlsx"


#%%

# Função para processar os arquivos que gerarão o banco de dados

def processar_arquivo(arquivo_entrada, arquivo_saida):
    # Ler o arquivo ignorando linhas em branco e as que começam com "locus"
    with open(arquivo_entrada, 'r') as file:
        linhas = [linha.strip() for linha in file if linha.strip() and not linha.startswith('Locus')]

    # Dividir as linhas em colunas e criar um DataFrame
    dados = [linha.split()[:-1] for linha in linhas]  # Remover a última coluna de cada linha
    df = pd.DataFrame(dados)
    
    # Excluir as colunas 0, 7 e a última coluna (8), sendo 8 a última
    df = df.drop(columns=[0], errors='ignore')  # Remove a coluna 0
    
    # Adicionar a coluna de contagem das linhas
    df.insert(0, 'allele_count', range(1, len(df) + 1))
    
    # Salvar o DataFrame em um arquivo Excel
    df.to_excel(arquivo_saida, index=False)


#%%

# Gerando os bancos de dados

processar_arquivo(allpop_previous, allpop_bd)

df = pd.read_excel(allpop_bd)

# Definir o cabeçalho, reordenar e excluir a última coluna
df.columns = ['contagem de alelos', 'Norte-Africanos', 'Europeus', 'Leste-Asiáticos', 'Africanos', 'Cryptic', 'Sul-Asiáticos', 'Nativo-Americanos']
nova_ordem = ['contagem de alelos', 'Africanos', 'Norte-Africanos', 'Europeus', 'Sul-Asiáticos', 'Leste-Asiáticos', 'Nativo-Americanos', 'Cryptic']
df = df[nova_ordem]
df = df.drop(columns=['Cryptic'], errors='ignore')
df.to_excel(allpop_bd, index=False)


processar_arquivo(nampop_previous, nampop_bd)

df = pd.read_excel(nampop_bd)

# Definir o cabeçalho, reordenar e excluir a última coluna
df.columns = ['contagem de alelos', 'Norte-Africanos', 'Europeus', 'Leste-Asiáticos', 'Africanos', 'Cryptic', 'Sul-Asiáticos', 'Nativo-Americanos']
nova_ordem = ['contagem de alelos', 'Africanos', 'Norte-Africanos', 'Europeus', 'Sul-Asiáticos', 'Leste-Asiáticos', 'Nativo-Americanos', 'Cryptic']
df = df[nova_ordem]
df = df.drop(columns=['Cryptic'], errors='ignore')
df.to_excel(nampop_bd, index=False)


#%%

# Especificando arquivos de saída

# Análise de clusteres
allpop_clusterizado = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\clusterizado_allpop_500MH.xlsx"  
nampop_clusterizado = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\clusterizado_nampop_500MH.xlsx"  

# Resumo, contagem de marcadores por cluster
count_allpop_nampop_resumo = "D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\count_allpop_nampop.xlsx" 

# Caminhos para salvar os heatmaps
heatmap_allpop = 'D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\heatmap_allpop.png'
heatmap_nampop = 'D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\heatmap_nampop.png'


#%%

# Função para clusterizar um arquivo de entrada, salvar a tabela clusterizada e elaborar heatmaps

def clusterizar_e_visualizar(arquivo_entrada, arquivo_saida_clusterizado, heatmap_saida, label_info):
    # Ler o arquivo Excel
    df = pd.read_excel(arquivo_entrada)
    
    # Selecionar as colunas de interesse para a clusterização (colunas 1 a 6)
    dados = df.iloc[:, 1:7]

    # Aplicar K-means para separar em 7 clusters
    kmeans = KMeans(n_clusters=7, random_state=100)
    df['Cluster'] = kmeans.fit_predict(dados)

    # Ordenar o DataFrame com base nos clusters para melhor visualização
    df_sorted = df.sort_values(by='Cluster')

    # Salvar o DataFrame com a coluna de clusters no Excel
    df_sorted.to_excel(arquivo_saida_clusterizado, index=False)

    # Plotar o heatmap dos dados clusterizados
    plt.figure(figsize=(12, 9))
    sns.heatmap(df_sorted.iloc[:, 1:7], cmap='viridis', cbar=True, 
                yticklabels=False, xticklabels=df.columns[1:7])
    plt.title(f'Clusterização K-means - Heatmap \n({label_info})', fontsize=18)
    plt.xlabel('Grupos Populacionais', fontsize=16)
    plt.ylabel('Alelos (Ordenados por Cluster)', fontsize=16)
    plt.xticks(fontsize=14,rotation=45)
    
    #plt.show()
        
    # Salvar o heatmap em 1200 DPI e formato PNG
    plt.savefig(heatmap_saida, dpi=1200, format='png', bbox_inches='tight')
    plt.close()  # Fecha a figura para liberar memória
    
    # Exibir as primeiras linhas da tabela com os clusters para visualização rápida
    print(f"Primeiras linhas do DataFrame com clusters ({label_info}):")
    print(df_sorted.head())

    # Contar as ocorrências de cada cluster
    contagem_clusters = df['Cluster'].value_counts().sort_index()
    
    # Criar um DataFrame de resumo com as contagens e o label de informação
    df_resumo = pd.DataFrame([[label_info] + contagem_clusters.tolist()], columns=['Informação'] + [f'Cluster {i}' for i in range(7)])
    
    return df_resumo
    
    
#%%

# Executar a função de clusterização

resumo_allpop = clusterizar_e_visualizar(allpop_bd, allpop_clusterizado, heatmap_allpop, "Informatividade Geral")
resumo_nampop = clusterizar_e_visualizar(nampop_bd, nampop_clusterizado, heatmap_nampop, "Informatividade Específica para Nativo-Americanos")


#%%

# Combinar os resumos em uma tabela única
df_resumo_final = pd.concat([resumo_allpop, resumo_nampop], ignore_index=True)

# Salvar o resumo dos clusters em um arquivo Excel único
df_resumo_final.to_excel(count_allpop_nampop_resumo, index=False)

# Exibir a tabela de resumo
print("\nResumo da contagem de amostras por cluster (Tabela Final):")
print(df_resumo_final)


#%% 

# Função para elaborar boxplots

def gerar_boxplots (arquivo_entrada, label_info, label_saida):
    df = pd.read_excel(arquivo_entrada)

    # Selecionar os dados (colunas 1 a 6)
    dados = df.iloc[:, 1:7]

    # Gerar boxplots para cada cluster
    clusters = df['Cluster'].unique()

    # Loop para criar um boxplot por cluster
    for cluster in clusters:
    # Filtrar os dados para o cluster atual
        cluster_data = df[df['Cluster'] == cluster]
    
        # Criar o boxplot para todas as colunas
        plt.figure(figsize=(12, 9))
        sns.boxplot(data=cluster_data[dados.columns])
        plt.title(f'Boxplot para o Cluster {cluster+1} \n({label_info})', fontsize=18)
        plt.xlabel('Grupos Populacionais', fontsize=16)
        plt.ylabel('Frequências Alélicas', fontsize=16)
        plt.xticks(fontsize=14,rotation=45)
        plt.yticks(fontsize=14)
        #plt.show()
        
        # Salvar boxplots em 1200 DPI e formato PNG
        plt.savefig(f'D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\boxplot_{label_saida}_cluster_{cluster+1}.png', dpi=1200,bbox_inches='tight')
        plt.close() 

#%%

gerar_boxplots(nampop_clusterizado, "Informatividade Específica para Nativo-Americanos", "nampop")
gerar_boxplots(allpop_clusterizado, "Informatividade Geral", "allpop")


#%%

# Função para verificar normalidade (de cada boxplot - Shapiro-Wilk) e homocedasticidade (Variâncias entre os grupos de cada cluster - Levene) e gerar uma tabela resumo com p-valores

def gerar_tabela_shapiro_wilk_levene(arquivo_entrada, label_info):
    df = pd.read_excel(arquivo_entrada)

    # Selecionar os dados (colunas 1 a 6)
    dados = df.iloc[:, 1:7]

    # Lista para armazenar os resultados dos p-valores
    resultados = []

    # Gerar testes para cada cluster
    clusters = df['Cluster'].unique()

    # Loop para cada cluster
    for cluster in clusters:
        # Filtrar os dados para o cluster atual
        cluster_data = df[df['Cluster'] == cluster]

        # Loop para cada coluna (grupo populacional)
        for col in dados.columns:
            # Teste de Shapiro-Wilk para normalidade
            stat_normalidade, p_value_normalidade = stats.shapiro(cluster_data[col])
            normalidade = 'Normal' if p_value_normalidade > 0.05 else 'Não Normal'  # Verifica a normalidade
            
            # Adicionar resultado à lista
            resultados.append({
                'Cluster': f'Cluster {cluster+1}',
                'Grupo Populacional': col,
                'Teste': 'Shapiro-Wilk (Normalidade)',
                'p-valor': p_value_normalidade,
                'Normalidade': normalidade,
                'Homocedasticidade': 'N/A'  # Não se aplica para este teste
            })

        # Teste de Levene para homogeneidade das variâncias
        stat_levene, p_value_levene = stats.levene(*[cluster_data[col] for col in dados.columns])
        homocedasticidade_resultado = 'Homogênea' if p_value_levene > 0.05 else 'Não Homogênea'
        
        # Adicionar resultado do Levene à lista (uma linha por cluster)
        resultados.append({
            'Cluster': f'Cluster {cluster+1}',
            'Grupo Populacional': 'Todos',
            'Teste': 'Levene (Homocedasticidade)',
            'p-valor': p_value_levene,
            'Normalidade': 'N/A',  # Não se aplica para este teste
            'Homocedasticidade': homocedasticidade_resultado
        })

    # Criar DataFrame com os resultados
    tabela_resumo = pd.DataFrame(resultados)

    # Exibir a tabela
    print(f"Tabela Resumo de p-valores ({label_info})")
    print(tabela_resumo)

    # Retornar a tabela caso queira exportar
    return tabela_resumo

#%%

# Executar a função para os arquivos
tabela_nampop = gerar_tabela_shapiro_wilk_levene(nampop_clusterizado, "Informatividade Específica para Nativo-Americanos")
tabela_allpop = gerar_tabela_shapiro_wilk_levene(allpop_clusterizado, "Informatividade Geral")

# Salvar as tabelas em arquivos Excel
tabela_nampop.to_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\tabela_shapiro_wilk_levene_nampop.xlsx", index=False)
tabela_allpop.to_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\tabela_shapiro_wilk_levene_allpop.xlsx", index=False)


#%%

# Função para realizar o teste de Kruskal-Wallis e salvar p-valores em uma tabela - verificar se há diferença entre os grupos de cada cluster
# Os dados não apresentam distribuição normal e não apresentam variâncias homogêneas

def gerar_tabela_kruskal_wallis(arquivo_entrada, label_info):
    df = pd.read_excel(arquivo_entrada)

    # Selecionar os dados (colunas 1 a 6)
    dados = df.iloc[:, 1:7]

    # Lista para armazenar os resultados dos p-valores
    resultados = []

    # Gerar Kruskal-Wallis para cada cluster
    clusters = df['Cluster'].unique()

    # Loop para cada cluster
    for cluster in clusters:
        # Filtrar os dados para o cluster atual
        cluster_data = df[df['Cluster'] == cluster]

        # Verificar se há mais de um grupo a ser comparado (necessário para Kruskal-Wallis)
        if len(cluster_data) > 1:
            # Teste de Kruskal-Wallis para as colunas (grupos populacionais)
            stat, p_value = stats.kruskal(*[cluster_data[col] for col in dados.columns])
            
            # Determinar se o teste de Dunn será necessário (se o p-valor < 0.05)
            dunn_necessario = 'Sim' if p_value < 0.05 else 'Não'
            
            # Adicionar resultado à lista
            resultados.append({
                'Cluster': f'Cluster {cluster+1}',
                'Grupos Populacionais': ', '.join(dados.columns),
                'p-valor Kruskal-Wallis': p_value,
                'Teste de Dunn Necessário': dunn_necessario
            })

    # Criar DataFrame com os resultados
    tabela_resumo = pd.DataFrame(resultados)

    # Exibir a tabela
    print(f"Tabela Resumo de p-valores Kruskal-Wallis ({label_info})")
    print(tabela_resumo)

    # Retornar a tabela para exportar
    return tabela_resumo


#%%

# Executar a função para o teste de Kruskal-Wallis para os arquivos
tabela_nampop = gerar_tabela_kruskal_wallis(nampop_clusterizado, "Informatividade Específica para Nativo-Americanos")
tabela_allpop = gerar_tabela_kruskal_wallis(allpop_clusterizado, "Informatividade Geral")

# Salvar as tabelas em arquivos Excel
tabela_nampop.to_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\tabela_kruskal_wallis_nampop.xlsx", index=False)
tabela_allpop.to_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\tabela_kruskal_wallis_allpop.xlsx", index=False)

#%%

# Função para realizar o teste de Dunn e salvar resultados em uma tabela - todos os resultados do Kruskal-Wallis indicam p<0.05

def realizar_teste_dunn(arquivo_entrada, label_info):
    df = pd.read_excel(arquivo_entrada)

    # Selecionar os dados (colunas 1 a 6)
    dados = df.iloc[:, 1:7]

    # Lista para armazenar os resultados dos testes
    resultados = []

    # Obter os clusters únicos
    clusters = df['Cluster'].unique()

    # Loop para cada cluster
    for cluster in clusters:
        # Filtrar os dados para o cluster atual
        cluster_data = df[df['Cluster'] == cluster]

        # Preparar dados para o teste de Dunn
        data = []
        for col in dados.columns:
            data.extend(cluster_data[col].tolist())
        groups = []
        for col in dados.columns:
            groups.extend([col] * len(cluster_data[col]))

        # Criar DataFrame para os resultados do Dunn
        dunn_results = pd.DataFrame({'Grupo': groups, 'Valores': data})

        # Realizar o teste de Dunn usando scikit_posthocs
        p_values = posthoc_dunn(dunn_results, val_col='Valores', group_col='Grupo', p_adjust='bonferroni')

        # Adicionar resultados à lista
        for i, grupo_a in enumerate(dados.columns):
            for j, grupo_b in enumerate(dados.columns):
                if i < j:  # Evitar comparações duplicadas
                    p_valor = p_values.loc[grupo_a, grupo_b]
                    resultados.append({
                        'Cluster': f'Cluster {cluster+1}',
                        'Grupo A': grupo_a,
                        'Grupo B': grupo_b,
                        'p-valor': p_valor,
                        'Diferentes': "Sim" if p_valor < 0.05 else "Não"  # Indica se os grupos são diferentes
                    })

    # Criar DataFrame com os resultados
    tabela_dunn = pd.DataFrame(resultados)

    # Exibir a tabela
    print(f"Tabela Resumo do Teste de Dunn ({label_info})")
    print(tabela_dunn)

    # Retornar a tabela para exportar
    return tabela_dunn


#%%

# Executar a função para o teste de Dunn para os arquivos
tabela_dunn_nampop = realizar_teste_dunn(nampop_clusterizado, "Informatividade Específica para Nativo-Americanos")
tabela_dunn_allpop = realizar_teste_dunn(allpop_clusterizado, "Informatividade Geral")

# Salvar as tabelas em arquivos Excel
tabela_dunn_nampop.to_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\tabela_dunn_nampop.xlsx", index=False)
tabela_dunn_allpop.to_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\tabela_dunn_allpop.xlsx", index=False)


#%%
# Caso queira visualizar os clusteres gerados em um gráfico 3D.

# Converter a coluna 'Cluster' para categórico

allpop_clusterizado = pd.read_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\clusterizado_allpop_500MH.xlsx")  
nampop_clusterizado = pd.read_excel("D:\\OneDrive\\BACKUP 2018\\documentos\\DSA - Esalq\\00 TCC\\dados\\clusterizado_nampop_500MH.xlsx")

allpop_clusterizado['Cluster'] = allpop_clusterizado['Cluster'].astype('category')
nampop_clusterizado['Cluster'] = nampop_clusterizado['Cluster'].astype('category')


# Definir paleta de cores categórica (usando o seaborn para obter as cores)
palette = sns.color_palette("Set1", n_colors=len(nampop_clusterizado['Cluster'].unique()))

# Criar um dicionário para mapear clusters a cores
cluster_colors = {cluster: palette[i] for i, cluster in enumerate(nampop_clusterizado['Cluster'].unique())}

# Iniciar a figura e os eixos 3D
fig = plt.figure(figsize=(6, 15))
ax = fig.add_subplot(111, projection='3d')

# Plotar os dados, atribuindo cores com base no cluster
for cluster in nampop_clusterizado['Cluster'].unique():
    cluster_data = nampop_clusterizado[nampop_clusterizado['Cluster'] == cluster]
    ax.scatter(cluster_data['Africanos'], 
               cluster_data['Europeus'], 
               cluster_data['Nativo-Americanos'], 
               label=f'Cluster {cluster}', 
               color=cluster_colors[cluster], 
               s=50)  # 's' é o tamanho dos pontos

# Configurar rótulos e título
ax.set_xlabel('Africanos')
ax.set_ylabel('Europeus')
ax.set_zlabel('Nativo-Americanos')
#ax.set_title('Gráfico 3D de Africanos, Norte-Africanos e Europeus (por Cluster)')

# Adicionar legenda
ax.legend(loc='upper left', bbox_to_anchor=(1, 0))

# Mostrar o gráfico
plt.show()

#%% FIM
