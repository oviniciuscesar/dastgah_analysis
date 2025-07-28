
"""
Análise Comparativa de Música Persa - VERSÃO DEFINITIVA COM ONSET DETECTION

Implementa a metodologia final usando detecção de onsets para uma segmentação
de notas robusta, seguida pela análise de similaridade com lógica fuzzy.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import hdbscan

# --------------------------------------------------------------------------
# DADOS TEÓRICOS (O "GABARITO")
# --------------------------------------------------------------------------
SHUR_TEORICO_INTERVALOS = np.array([0, 149, 300, 500, 702, 783, 985])
# SHUR_TEORICO_INTERVALOS = np.array([0, 149, 300, 500])
# SHUR_TEORICO_PESOS = np.array([0.77, 0.96, 0.10, 0.36])
SHUR_TEORICO_PESOS = np.array([0.77, 0.96, 0.10, 0.36, 0.80, 0.30, 0.10])
SHUR_GRAUS_LABELS = ['Tônica (Shur)', '2º Grau (149c)', '3º Grau (300c)', '4º Grau (500c)', '5º Grau (702c)', '6º Grau (783c)', '7º Grau (985c)']

# --------------------------------------------------------------------------
# FUNÇÕES DE ANÁLISE
# --------------------------------------------------------------------------

def extrair_sinal_e_pitch(audio_path):
    """Carrega áudio e extrai tanto o sinal bruto (y) quanto o contorno de pitch (f0)."""
    print(f"Carregando e processando áudio: {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=44100)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
        times = librosa.times_like(f0)
        return y, sr, f0, times
    except Exception as e:
        print(f"Erro ao processar o arquivo {audio_path}: {e}")
        return None, None, None, None

def segmentar_notas_por_onset(y, sr, f0, times, ref_hz):
    """
    Segmenta notas usando Detecção de Onsets.
    """
    print("Detectando inícios de notas (onsets)...")
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, wait=0.05, pre_avg=0.05, post_avg=0.05)
    onset_frames = librosa.time_to_frames(onset_times, sr=sr)
    onset_frames = np.concatenate([[0], onset_frames, [len(f0) - 1]])
    
    print(f"DEBUG: {len(onset_times)} onsets detectados.")
    
    analise = []
    final_labels = np.full_like(f0, -1, dtype=int)
    note_centers = []

    MIN_PONTOS_NOTA = 10 #frames de áudio com afinação estável
    MAX_ESTABILIDADE = 50 # desvio padrão máximo em Cents permitido para considerar uma nota válida

    nota_valida_idx = 0
    for i in range(len(onset_frames) - 1):
        start_frame, end_frame = onset_frames[i], onset_frames[i+1]
        segmento_f0_voiced = f0[start_frame:end_frame][~np.isnan(f0[start_frame:end_frame])]
        
        if len(segmento_f0_voiced) < MIN_PONTOS_NOTA:
            continue
            
        segmento_cents = 1200 * np.log2(segmento_f0_voiced / ref_hz)
        stability = np.std(segmento_cents)
        
        if stability < MAX_ESTABILIDADE:
            media = np.mean(segmento_cents)
            analise.append({
                "Nota ID": nota_valida_idx,
                "Afinação Média (Cents)": media,
                "Estabilidade (Desvio Padrão em Cents)": stability,
                "Ocorrências": len(segmento_cents)
            })
            final_labels[start_frame:end_frame] = len(note_centers)
            note_centers.append(media)
            nota_valida_idx += 1

    print(f"DEBUG: {len(analise)} notas válidas encontradas após a validação.")
    
    df_analise = pd.DataFrame(analise)
    contorno_cents = 1200 * np.log2(f0 / ref_hz)
    mask_validos = np.isin(final_labels, list(range(len(note_centers))))
    final_labels[~mask_validos] = -1
    contorno = {'times': librosa.times_like(f0), 'cents': contorno_cents, 'labels': final_labels}
    
    return df_analise, contorno, np.array(sorted(note_centers))

def quantizar_escala_performada(df_performance, tonica_performance):
    """
    Mapeia todas as notas detectadas para as 7 classes do Dastgah teórico.
    RETORNA: O DataFrame agregado E o DataFrame original com a coluna 'classe_teorica'.
    """
    if df_performance.empty:
        return pd.DataFrame(), pd.DataFrame()

    shur_teorico_transposto = SHUR_TEORICO_INTERVALOS + tonica_performance
    
    # Atribui cada nota performada à classe teórica mais próxima
    distancias = np.abs(df_performance['Afinação Média (Cents)'].values[:, np.newaxis] - shur_teorico_transposto)
    df_performance_com_classes = df_performance.copy()
    df_performance_com_classes['classe_teorica'] = np.argmin(distancias, axis=1)
    
    # Agrupa as notas por classe para criar o DataFrame agregado
    dados_quantizados = []
    for i in range(len(shur_teorico_transposto)):
        notas_da_classe = df_performance_com_classes[df_performance_com_classes['classe_teorica'] == i]
        
        if not notas_da_classe.empty:
            media_afinacao = np.average(notas_da_classe['Afinação Média (Cents)'], weights=notas_da_classe['Ocorrências'])
            media_estabilidade = np.average(notas_da_classe['Estabilidade (Desvio Padrão em Cents)'], weights=notas_da_classe['Ocorrências'])
            total_ocorrencias = notas_da_classe['Ocorrências'].sum()
            
            dados_quantizados.append({
                "Grau": i,
                "Afinação Média (Cents)": media_afinacao,
                "Estabilidade (Desvio Padrão em Cents)": media_estabilidade,
                "Ocorrências": total_ocorrencias
            })
            
    df_agregado = pd.DataFrame(dados_quantizados)
    
    return df_agregado, df_performance_com_classes


def calcular_similaridade_com_teoria(df_quantizado, tonica_performance):
    """
    Calcula a similaridade final usando um 'Peso Contextual' que considera
    tanto o peso teórico da nota quanto sua frequência de ocorrência na performance.
    """
    universo = np.arange(np.floor(tonica_performance) - 500, np.ceil(tonica_performance) + 1500, 1.0)
    
    # Fuzzifica as notas teóricas
    shur_teorico_transposto = SHUR_TEORICO_INTERVALOS + tonica_performance
    fuzzy_notes_teoricas = [fuzzify_note(nota, 0, universo, is_theoretical=True) for nota in shur_teorico_transposto]

    # Fuzzifica as notas já quantizadas da performance
    fuzzy_notes_performance = {
        int(row['Grau']): fuzzify_note(row['Afinação Média (Cents)'], row['Estabilidade (Desvio Padrão em Cents)'], universo)
        for _, row in df_quantizado.iterrows()
    }

    # Compara diretamente classe por classe
    similarities = []
    for i in range(len(fuzzy_notes_teoricas)):
        if i in fuzzy_notes_performance:
            sim = jaccard_similarity(fuzzy_notes_teoricas[i], fuzzy_notes_performance[i])
            similarities.append(sim)
        else:
            similarities.append(0.0)

    # --- NOVA LÓGICA DE PONDERAÇÃO CONTEXTUAL ---
    # 1. Cria um array de ocorrências na mesma ordem dos graus teóricos
    ocorrencias = np.zeros(len(SHUR_TEORICO_INTERVALOS))
    for _, row in df_quantizado.iterrows():
        grau_idx = int(row['Grau'])
        if grau_idx < len(ocorrencias):
            ocorrencias[grau_idx] = row['Ocorrências']
            
    # 2. Calcula o novo Peso Contextual
    pesos_teoricos = SHUR_TEORICO_PESOS
    pesos_contextuais = pesos_teoricos * ocorrencias
    
    # 3. Calcula a Média Ponderada Fuzzy (FWA) usando os novos pesos
    numerador = np.sum(np.array(similarities) * pesos_contextuais)
    denominador = np.sum(pesos_contextuais)
    
    fwa_score = numerador / denominador if denominador > 0 else 0.0
    # --- FIM DA NOVA LÓGICA ---
    
    return fwa_score, similarities

def fuzzify_note(mean_cent, std_cent, universe, is_theoretical=False):
    std_umf, std_lmf = (20.0, 10.0) if is_theoretical else (std_cent + 5.0, max(1.0, std_cent - 5.0))
    umf = fuzz.gaussmf(universe, mean_cent, std_umf)
    lmf = fuzz.gaussmf(universe, mean_cent, std_lmf)
    return {'umf': umf, 'lmf': lmf}

def jaccard_similarity(it2fs_A, it2fs_B):
    num = np.sum(np.minimum(it2fs_A['umf'], it2fs_B['umf'])) + np.sum(np.minimum(it2fs_A['lmf'], it2fs_B['lmf']))
    den = np.sum(np.maximum(it2fs_A['umf'], it2fs_B['umf'])) + np.sum(np.maximum(it2fs_A['lmf'], it2fs_B['lmf']))
    return num / den if den > 0 else 0.0

# def analisar_fase_3_padroes_melodicos(df_performance_com_classes):
#     """
#     FASE 3: Analisa a sequência de notas para encontrar padrões melódicos específicos.
#     (Versão corrigida para aceitar o argumento correto)
#     """
#     if df_performance_com_classes is None or df_performance_com_classes.empty or 'classe_teorica' not in df_performance_com_classes.columns:
#         return {"total_repousos_tonica": 0, "articulacoes_pelo_2_grau": 0, "taxa_articulacao": 0}

#     # Passo 1: Extrair a sequência de classes de notas
#     sequencia_de_classes = df_performance_com_classes['classe_teorica'].tolist()
#     duracoes = df_performance_com_classes['Ocorrências'].tolist()

#     # Passo 2: Definir o que é uma "nota longa"
#     if not duracoes:
#         return {"total_repousos_tonica": 0, "articulacoes_pelo_2_grau": 0, "taxa_articulacao": 0}
#     limiar_longa = np.percentile(duracoes, 75)

#     # Passo 3: Buscar o padrão e contar
#     total_repousos_tonica = 0
#     articulacoes_pelo_2_grau = 0
#     for i in range(1, len(sequencia_de_classes)):
#         if sequencia_de_classes[i] == 0 and duracoes[i] >= limiar_longa:
#             total_repousos_tonica += 1
#             if sequencia_de_classes[i-1] == 1:
#                 articulacoes_pelo_2_grau += 1
    
#     # Passo 4: Calcular a taxa de articulação
#     taxa_articulacao = (articulacoes_pelo_2_grau / total_repousos_tonica) * 100 if total_repousos_tonica > 0 else 0
    
#     return {
#         "total_repousos_tonica": total_repousos_tonica,
#         "articulacoes_pelo_2_grau": articulacoes_pelo_2_grau,
#         "taxa_articulacao": round(taxa_articulacao, 2)
#     }

def analisar_fase_3_padroes_melodicos(df_performance_com_classes):
    """
    FASE 3: Analisa a sequência de notas para encontrar padrões melódicos,
    calculando duas métricas distintas para a articulação da tônica.
    """
    if df_performance_com_classes is None or df_performance_com_classes.empty or 'classe_teorica' not in df_performance_com_classes.columns:
        # Retorna um dicionário com a estrutura de resultados zerada
        return {
            "geral_total_tonicas": 0, "geral_articulacoes": 0, "taxa_articulacao_geral": 0,
            "repouso_total_tonicas": 0, "repouso_articulacoes": 0, "taxa_articulacao_repouso": 0
        }

    sequencia_de_classes = df_performance_com_classes['classe_teorica'].tolist()
    duracoes = df_performance_com_classes['Ocorrências'].tolist()

    if not duracoes:
        # Estrutura de resultados zerada
        return {
            "geral_total_tonicas": 0, "geral_articulacoes": 0, "taxa_articulacao_geral": 0,
            "repouso_total_tonicas": 0, "repouso_articulacoes": 0, "taxa_articulacao_repouso": 0
        }

    limiar_longa = np.percentile(duracoes, 75)

    # Contadores para a análise GERAL
    geral_total_tonicas = sequencia_de_classes.count(0)
    geral_articulacoes = 0
    
    # Contadores para a análise de REPOUSO
    repouso_total_tonicas = 0
    repouso_articulacoes = 0

    for i in range(1, len(sequencia_de_classes)):
        # Análise Geral: verifica todas as tônicas
        if sequencia_de_classes[i] == 0:
            if sequencia_de_classes[i-1] == 1:
                geral_articulacoes += 1
        
        # Análise de Repouso: verifica apenas as tônicas longas
        if sequencia_de_classes[i] == 0 and duracoes[i] >= limiar_longa:
            repouso_total_tonicas += 1
            if sequencia_de_classes[i-1] == 1:
                repouso_articulacoes += 1
    
    # Calcula as taxas
    taxa_geral = (geral_articulacoes / geral_total_tonicas) * 100 if geral_total_tonicas > 0 else 0
    taxa_repouso = (repouso_articulacoes / repouso_total_tonicas) * 100 if repouso_total_tonicas > 0 else 0
    
    return {
        "geral_total_tonicas": geral_total_tonicas,
        "geral_articulacoes": geral_articulacoes,
        "taxa_articulacao_geral": round(taxa_geral, 2),
        "repouso_total_tonicas": repouso_total_tonicas,
        "repouso_articulacoes": repouso_articulacoes,
        "taxa_articulacao_repouso": round(taxa_repouso, 2)
    }

def imprimir_tabela_comparativa(res_humano, res_ia):
    print("\n" + "="*80 + "\nTABELA COMPARATIVA FINAL\n" + "="*80)
    df = pd.DataFrame({
        'Grau': [label.split('\n')[0] for label in SHUR_GRAUS_LABELS],
        'Intervalo Teórico (c)': SHUR_TEORICO_INTERVALOS,
        'Peso Teórico': SHUR_TEORICO_PESOS,
        'Similaridade Humano': res_humano.get('similaridades', [0]*7),
        'Similaridade IA': res_ia.get('similaridades', [0]*7)
    }).set_index('Grau')
    formatters = {'Intervalo Teórico (c)': "{:.2f}".format, 'Peso Teórico': "{:.2f}".format,
                  'Similaridade Humano': "{:.3f}".format, 'Similaridade IA': "{:.3f}".format}
    print(df.to_string(formatters=formatters, na_rep='-'))
    print("\n--- Pontuações Finais (Média Ponderada Fuzzy) ---")
    print(f"Score de Similaridade Humano: {res_humano.get('score', 0.0):.4f}")
    print(f"Score de Similaridade IA:     {res_ia.get('score', 0.0):.4f}")
    
    print("\n" + "="*80 + "\nCOMPARAÇÃO DA ESTRUTURA DA ESCALA (RELATIVA À TÔNICA)\n" + "="*80)
    
    max_len = max(len(SHUR_TEORICO_INTERVALOS), len(res_humano.get('escala_relativa_cents', [])), len(res_ia.get('escala_relativa_cents', [])))
    
    def pad_list(lst, length):
        return lst + [np.nan] * (length - len(lst))

    data = {
        'Grau': pad_list([f'{i+1}º' for i in range(max_len)], max_len),
        'Teoria (Shur)': pad_list(list(SHUR_TEORICO_INTERVALOS), max_len),
        'Humano (Extraído)': pad_list(res_humano.get('escala_relativa_cents', []), max_len),
        'IA (Extraído)': pad_list(res_ia.get('escala_relativa_cents', []), max_len)
    }

    df_esc = pd.DataFrame(data).set_index('Grau')
    formatters_esc = {col: "{:.2f}".format for col in df_esc.columns}
    print(df_esc.to_string(formatters=formatters_esc, na_rep='-'))
    print("="*80)


def plotar_dashboard_completo(res_humano, res_ia):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.2]})
    fig.suptitle('Análise Comparativa Dastgah Shur: Performance Humana vs. Geração por IA', fontsize=20)

    def plot_contorno(ax, res, tipo):
        contorno = res.get('contorno', {})
        tonica = res.get('tonica', 0)
        df_com_classes = res.get('df_com_classes')

        if not isinstance(contorno, dict) or df_com_classes is None:
            ax.text(0.5, 0.5, f'Dados de contorno para "{tipo}" indisponíveis.', ha='center', va='center')
            return
        
        # Cria mapa de ID da nota detectada -> classe teórica (0-6)
        mapa_id_para_classe = df_com_classes.set_index('Nota ID')['classe_teorica'].to_dict()
        
        # Mapeia os labels do contorno (que são IDs) para as classes teóricas
        labels_por_classe = np.array([mapa_id_para_classe.get(l, -1) for l in contorno['labels']])
        
        # Plotagem
        noise_mask = (labels_por_classe == -1)
        ax.plot(contorno['times'], contorno['cents'], '.', markersize=2, color='gray', alpha=0.3)
        
        unique_labels = sorted([l for l in set(labels_por_classe) if l != -1])
        colors = plt.cm.viridis(np.linspace(0, 1, 7))
        
        for k in unique_labels:
            note_mask = (labels_por_classe == k)
            ax.plot(contorno['times'][note_mask], contorno['cents'][note_mask], '.', markersize=5, color=colors[k])
        
        shur_teorico_transposto = SHUR_TEORICO_INTERVALOS + tonica
        for i, center in enumerate(shur_teorico_transposto):
             ax.axhline(center, color='red', linestyle='--', linewidth=1.0, alpha=0.9)

        ax.set_title(f'Contorno Melódico ({tipo})')
        ax.set_ylabel('Intervalo (Cents)')
        ax.set_xlabel('Tempo (s)')
        # ax.legend(['Ruído/Transição'], loc='upper right')
        ax.grid(True, linestyle=':')

    plot_contorno(axs[0, 0], res_humano, "Humano")
    plot_contorno(axs[0, 1], res_ia, "IA")
    
    # --- Painel de Intervalos (RESTAURADO) ---
    ax3 = axs[1, 0]
    intervalos_h = res_humano.get('intervalos_quantizados', [])
    intervalos_ia = res_ia.get('intervalos_quantizados', [])
    if intervalos_h or intervalos_ia:
        n_intervalos = max(len(intervalos_h), len(intervalos_ia))
        index = np.arange(n_intervalos)
        bar_width = 0.35
        intervalos_h_padded = intervalos_h + [0] * (n_intervalos - len(intervalos_h))
        intervalos_ia_padded = intervalos_ia + [0] * (n_intervalos - len(intervalos_ia))
        bars1_int = ax3.bar(index - bar_width/2, intervalos_h_padded, bar_width, label='Humano', color='deepskyblue')
        bars2_int = ax3.bar(index + bar_width/2, intervalos_ia_padded, bar_width, label='IA', color='orangered')
        ax3.set_title('Estrutura Intervalar (Cents)')
        ax3.set_xlabel('Sequência de Notas')
        ax3.set_ylabel('Intervalo (Cents)')
        ax3.set_xticks(index)
        ax3.set_xticklabels([f'{i+1}º' for i in range(n_intervalos)])
        ax3.legend()
        ax3.grid(axis='y', linestyle=':')
    else:
        ax3.text(0.5, 0.5, 'Dados de intervalo indisponíveis.', ha='center', va='center')

    # --- Painel de Similaridade Fuzzy ---
    ax4 = axs[1, 1]
    n_graus = len(SHUR_GRAUS_LABELS)
    index_sim = np.arange(n_graus)
    bar_width_sim = 0.35
    bars3_sim = ax4.bar(index_sim - bar_width_sim/2, res_humano.get('similaridades', [0]*n_graus), bar_width_sim, label=f"Humano (Similaridade total: {res_humano.get('score', 0):.2f})", color='deepskyblue')
    bars4_sim = ax4.bar(index_sim + bar_width_sim/2, res_ia.get('similaridades', [0]*n_graus), bar_width_sim, label=f"IA (Similaridade total: {res_ia.get('score', 0):.2f})", color='orangered')
    ax4.set_title('Similaridade com o Dastgah Shur')
    ax4.set_ylabel('Pontuação de Similaridade Fuzzy (JSM)')
    ax4.set_xticks(index_sim)
    ax4.set_xticklabels(SHUR_GRAUS_LABELS, rotation=45, ha="right")
    ax4.set_ylim(0, 1.05)
    ax4.legend()
    ax4.grid(axis='y', linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("analise_completa.png", dpi=300)
    print("\nAnálise completa salva como 'analise_completa.png'")
    plt.show()


# --- EXECUÇÃO DO SCRIPT ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

# #     # Define os caminhos para os seus arquivos de áudio de forma robusta
    audio_humano = os.path.join(script_dir, 'DastgahShur.wav')
    audio_ia = os.path.join(script_dir, 'DastgahShurIA.wav')
    resultados_humano, resultados_ia = {}, {}

    for tipo, caminho in [("Humano", audio_humano), ("IA", audio_ia)]:
        print("\n" + "="*80 + f"\nINICIANDO ANÁLISE: {tipo.upper()}\n" + "="*80)
        y, sr, f0, times = extrair_sinal_e_pitch(caminho)
        if f0 is not None:
            df_performance, contorno, centros = segmentar_notas_por_onset(y, sr, f0, times, librosa.note_to_hz('C0'))
            if df_performance is not None and not df_performance.empty:
                tonica = df_performance.loc[df_performance['Ocorrências'].idxmax()]['Afinação Média (Cents)']
                df_quantizado, df_com_classes = quantizar_escala_performada(df_performance, tonica)
                score, sims = calcular_similaridade_com_teoria(df_quantizado, tonica)
                
                intervalos_q = []
                escala_relativa = []
                if not df_quantizado.empty:
                    notas_ordenadas = sorted(df_quantizado['Afinação Média (Cents)'])
                    tonica_quantizada = notas_ordenadas[0]
                    # Calcula os intervalos sucessivos
                    intervalos_q = list(np.diff(notas_ordenadas))
                    # Calcula a escala relativa à tônica
                    escala_relativa = [nota - tonica_quantizada for nota in notas_ordenadas]

                # Dicionário de resultados completo para cada performance
                resultado = {
                    'tonica': tonica, 
                    'contorno': contorno, 
                    'centros_notas': centros,
                    'intervalos_quantizados': intervalos_q,
                    'escala_relativa_cents': escala_relativa, # Chave adicionada
                    'score': score, 
                    'similaridades': sims,
                    'df_quantizado': df_quantizado,
                    'df_com_classes': df_com_classes,
                }
                
                if tipo == "Humano":
                    resultados_humano = resultado
                else:
                    resultados_ia = resultado
    
    if resultados_humano and resultados_ia:
        imprimir_tabela_comparativa(resultados_humano, resultados_ia)
        plotar_dashboard_completo(resultados_humano, resultados_ia)

        # --- Execução da FASE 3 com os dados corretos ---
        print("\n" + "="*80)
        print("FASE 3: ANÁLISE DE PADRÕES MELÓDICOS (SINTAXE)")
        print("="*80)
        
        # Passa o DataFrame que contém a coluna 'classe_teorica'
        padroes_humano = analisar_fase_3_padroes_melodicos(resultados_humano['df_com_classes'])
        padroes_ia = analisar_fase_3_padroes_melodicos(resultados_ia['df_com_classes'])

        print("\n--- Análise da Articulação do Repouso na Tônica ---")
        print(f"Humano:")
        print(f"  - Análise de Repouso: Encontrados {padroes_humano['repouso_total_tonicas']} repousos longos na tônica.")
        print(f"    Destes, {padroes_humano['repouso_articulacoes']} foram articulados pelo 2º grau.")
        print(f"    >> Taxa de Articulação de REPOUSO (Humano): {padroes_humano['taxa_articulacao_repouso']}%")
        print(f"  - Análise Geral: Das {padroes_humano['geral_total_tonicas']} vezes que a tônica foi tocada, {padroes_humano['geral_articulacoes']} foram precedidas pelo 2º grau.")
        print(f"    >> Taxa de Articulação GERAL (Humano): {padroes_humano['taxa_articulacao_geral']}%")

        print(f"\nIA:")
        print(f"  - Análise de Repouso: Encontrados {padroes_ia['repouso_total_tonicas']} repousos longos na tônica.")
        print(f"    Destes, {padroes_ia['repouso_articulacoes']} foram articulados pelo 2º grau.")
        print(f"    >> Taxa de Articulação de REPOUSO (IA): {padroes_ia['taxa_articulacao_repouso']}%")
        print(f"  - Análise Geral: Das {padroes_ia['geral_total_tonicas']} vezes que a tônica foi tocada, {padroes_ia['geral_articulacoes']} foram precedidas pelo 2º grau.")
        print(f"    >> Taxa de Articulação GERAL (IA): {padroes_ia['taxa_articulacao_geral']}%")
        print("="*80)

    else:
        print("\nNão foi possível gerar os resultados pois uma das análises falhou.")