import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def score_realizado_por_meta(df, meta_atributo, realizado_atributo, rotulo):
    import pandas as pd
    import numpy as np
    score = []
    index = []

    for i in df.index:
        meta = df.loc[i, meta_atributo]
        indicador = df.loc[i, realizado_atributo]

        if np.isnan(indicador):
            continue
        elif float(indicador) > 0:
            index.append(i)
            score.append(indicador / meta)
        elif float(indicador) == 0:
            index.append(i)
            score.append(indicador)

    score_att = pd.Series(data=score, index=index)
    score_att = score_att.to_frame()
    score_att.columns = [rotulo]
    return score_att


def converte_horas_minutos(hora):
    """
    Converte horas em minutos. Feito para tratar horas negativas, quando estas estão em formato de strings
    :param hora: horas em formato de datetime.time ou em str
    :return: horas em minutos
    """
    from datetime import time

    if type(hora) is str:
        try:
            hora_componentes = hora.split(":")
            horas = int(hora_componentes[0]) * 60
            min = int(hora_componentes[1])
            return horas - min
        except ValueError:
            raise ValueError(
                "Valor não reconhecido pela função. Remova qualquer valor dos dados que não seja uma string no formato '-00:00:00' ou um tipo datetime.time")
        #     print("Mensagem original:", sys.exc_info()[1])
        #     sys.exit(0)
        # raise:
    elif type(hora) is time:
        return (hora.hour * 60) + hora.minute
    else:
        return hora

def transform_cpnj_format(cnpj):
    cnpj = str(cnpj)
    if len(cnpj) < 14:
        cnpj = cnpj.zfill(14)
    return "%s.%s.%s/%s-%s" % ( cnpj[0:2], cnpj[2:5], cnpj[5:8], cnpj[8:12], cnpj[12:14] )

def transform_cpf_format(cpf):
    cpf = str(cpf)
    if len(cpf) < 11:
        cpf = cpf.zfill(11)
    return f'{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}'


def tratar_atributos_com_pontuacao_categoria(s_raciocionio_adaptativo):
    """
    Recebe os valores do atributo Raciocionio (Adaptativo)-theta, no formato de 'valor real / valor categorico',
    e remove a barra e o valor categórico
    :param s_raciocionio_adaptativo: Series com valores do racíocinio adaptativo
    :return: lista com os novos valores do racíocinio
    """
    raciciocinio_adaptativo = []
    for i in s_raciocionio_adaptativo:
        if not pd.isna(i):
            x = str(i).split("/")
            raciciocinio_adaptativo.append(x[0].strip())
        else:
            raciciocinio_adaptativo.append(i)

    raciciocinio_adaptativo = pd.Series(raciciocinio_adaptativo,
                                        index=s_raciocionio_adaptativo.index)
    return raciciocinio_adaptativo


def unir_raciocinios(df, col_principal, col_secundaria):
    """
    Recebe um data frame com os dados do mindmatch e une os valores de Raciocínio com de Raciocíno Adaptativo
    (este já tratado, apenas com valores reais). De forma que quando Raciocínio não tiver valor, usar o Adaptativo
    no lugar.
    :param df: data frame com os dados do mindmatch
    :return: series com os valores unidos
    """
    raciocinio_preenchido = []
    for i in df.index:
        if pd.isna(df.at[i, col_principal]):
            raciocinio_preenchido.append(df.at[i, col_secundaria])
        else:
            raciocinio_preenchido.append(df.at[i, col_principal])

    return pd.Series(raciocinio_preenchido, index=df.index)


def analise_univariada(df, atributo, rotulo, positivo=1):
    labels = pd.qcut(df[atributo], 3, labels=["Q1", "Q2", "Q3"])

    analise_univariada = df.join(labels, rsuffix="_quantile")

    data = analise_univariada.query(
        rotulo + "==" + str(positivo)
    )[atributo + "_quantile"].value_counts().reset_index()

    data["porcentagem"] = round((data[atributo + "_quantile"] / data[atributo + "_quantile"].sum() * 100), 2)

    plt.figure(figsize=(5, 3))
    ax = sns.barplot(data=data,
                     x="index",
                     y=atributo + "_quantile")

    data = data.sort_values(by="index", ignore_index=True)

    ax.set_ylabel("Quantidade")
    ax.set_xlabel(atributo)

    print(data)
    ax.text(0, 2, str(data.loc[0, "porcentagem"]) + "%", color='black', style="oblique", ha="center")
    ax.text(1, 2, str(data.loc[1, "porcentagem"]) + "%", color='black', style="oblique", ha="center")
    ax.text(2, 2, str(data.loc[2, "porcentagem"]) + "%", color='black', style="oblique", ha="center")

    plt.show()


def analise_univariada_duas_barras(df, atributo, rotulo):
    labels = pd.qcut(df[atributo], 3, duplicates="drop")

    analise_univariada = df.join(labels, rsuffix="_quantile")

    data = analise_univariada.query(
        rotulo + "== 1"
    )[atributo + "_quantile"].value_counts().reset_index()
    data["porcentagem_perf"] = round((data[atributo + "_quantile"] / data[atributo + "_quantile"].sum() * 100), 2)

    data2 = analise_univariada.query(
        rotulo + "== 0"
    )[atributo + "_quantile"].value_counts().reset_index()
    data2["porcentagem_np"] = round((data2[atributo + "_quantile"] / data2[atributo + "_quantile"].sum() * 100), 2)

    data = data.set_index("index").join(data2.set_index("index"), rsuffix="_np")
    data.reset_index(inplace=True)
    data.sort_values(by="index", inplace=True)
    # print(data)
    labels = list(data["index"])
    porcentagem_perf = data.porcentagem_perf
    porcentagem_np = data.porcentagem_np

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rects1 = ax.bar(x - width / 2, porcentagem_perf, width, label='Performa')
    rects2 = ax.bar(x + width / 2, porcentagem_np, width, label='Não Performa')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('%')
    ax.set_title(f'{atributo} - Porcentagem perf. e N_perf. por quantile')
    ax.set_xticks(x)
    ax.set_ylim([0, 60])
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, label_type="center")
    ax.bar_label(rects2, padding=3, label_type="center")

    fig.tight_layout()

    plt.show()


def analise_univariada_diferenca(df, atributo, rotulo):
    labels = pd.qcut(df[atributo], 3)

    analise_univariada = df.join(labels, rsuffix="_quantile")

    data = analise_univariada.query(
        rotulo + "== 1"
    )[atributo + "_quantile"].value_counts().reset_index()
    data["porcentagem_perf"] = round((data[atributo + "_quantile"] / data[atributo + "_quantile"].sum() * 100), 2)

    data2 = analise_univariada.query(
        rotulo + "== 0"
    )[atributo + "_quantile"].value_counts().reset_index()
    data2["porcentagem_np"] = round((data2[atributo + "_quantile"] / data2[atributo + "_quantile"].sum() * 100), 2)

    data = data.set_index("index").join(data2.set_index("index"), rsuffix="_np")
    data.reset_index(inplace=True)
    data.sort_values(by="index", inplace=True)

    data["diff"] = (data["porcentagem_np"] - data["porcentagem_perf"]).abs()
    # print(data)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=data,
                     x="index",
                     y="diff",
                     color='green')

    data = data.sort_values(by="index", ignore_index=True)

    ax.set_ylabel("%")
    ax.set_xlabel(atributo)
    ax.set_ylim([0, 20])
    ax.set_title(f'{atributo} - diferença de porcentagem perf. e nao_perf.')
    ax.text(0, 1, f"{data.loc[0, 'diff']:.2f}%", color='black', style="oblique", ha="center")
    ax.text(1, 1, f"{data.loc[1, 'diff']:.2f}%", color='black', style="oblique", ha="center")
    ax.text(2, 1, f"{data.loc[2, 'diff']:.2f}%", color='black', style="oblique", ha="center")

    plt.show()


def get_tempo_vinculo_ativos(df: pd.DataFrame, nome_atributo_data: str, tipo_tempo: str = 'M') -> pd.Series:
    from datetime import datetime

    hoje = datetime.today()
    return ((hoje - df[nome_atributo_data]) / np.timedelta64(1, tipo_tempo)).astype(int)

def get_tempo_vinculo(df: pd.DataFrame,  admissao: str, demissao: str, tipo_tempo: str = 'M') -> pd.Series:
    from datetime import datetime

    hoje = datetime.today()
    return ((df[demissao] - df[admissao]) / np.timedelta64(1, tipo_tempo)).astype(int)

def get_medias_por_constructos(df) -> pd.DataFrame():
    import pandas as pd
    df_retorno = pd.DataFrame()
    df_retorno['media_mental'] = df[
        ['perfil-Capacidade analítica', 'perfil-Pensamento conceitual', 'perfil-Reflexão', 'perfil-Pensamento criativo',
         'perfil-Planejamento e organização']].mean(axis=1)

    df_retorno['media_social'] = df[['perfil-Comunicação',
                                         'perfil-Consideração pelos outros',
                                         'perfil-Influência',
                                         'perfil-Sociabilidade',
                                         'perfil-Facilitação',
                                         'perfil-Flexibilidade']].mean(axis=1)

    df_retorno['media_motivacional'] = df[['perfil-Estabilidade emocional',
                                               'perfil-Ambição',
                                               'perfil-Iniciativa',
                                               'perfil-Assertividade',
                                               'perfil-Tomada de riscos']].mean(axis=1)

    return df_retorno
