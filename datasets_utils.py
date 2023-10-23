
def get_colunas_mindmatch():
    '''
    Retorna os atributos (colunas) utilizados para análise do Mindmatch.
    :return: lista de strings com o nome dos atributos.
    '''
    return [
        'Raciocínio', 'Social', 'Motivacional', 'Cultura pontuação',
        'perfil-Capacidade analítica',
        'perfil-Pensamento conceitual', 'perfil-Reflexão',
        'perfil-Pensamento criativo', 'perfil-Planejamento e organização',
        'perfil-Comunicação', 'perfil-Consideração pelos outros',
        'perfil-Influência', 'perfil-Sociabilidade', 'perfil-Facilitação',
        'perfil-Flexibilidade', 'perfil-Estabilidade emocional',
        'perfil-Ambição', 'perfil-Iniciativa', 'perfil-Assertividade',
        'perfil-Tomada de riscos']


def rotulacao_performadores(atributo,
                            top_percentil=.75,
                            bottom_percentil=.25,
                            crescente=True):
    """
    Dado os valores de um atributo (list-like), seleciona os top performadores e bottom performadores, ou seja,
    os melhores e os piores dado o
    :param atributo:
    :param top_percentil:
    :param bottom_percentil:
    :param crescente:
    :return:
    """
    import pandas as pd

    quantis = atributo.quantile([top_percentil, bottom_percentil])
    print(quantis.iloc[0])
    print(quantis.iloc[1])
    if crescente:
        labels_top = (atributo >= quantis.iloc[0])
        labels_bottom = (atributo <= quantis.iloc[1])
    else:
        labels_bottom = (atributo >= quantis.iloc[0])
        labels_top = (atributo <= quantis.iloc[1])

    labels_top = labels_top.map(lambda x: 1 if x else 0)
    labels_bottom = labels_bottom.map(lambda x: 1 if x else 0)
    labels_top = pd.Series(labels_top, index= atributo.index)
    labels_bottom = pd.Series(labels_bottom, index=atributo.index)
    return labels_top, labels_bottom

def add_rotulos_dataframe(df, rotulos, nome_coluna_rotulo="rotulos"):
    df[nome_coluna_rotulo] = rotulos


def selecionar_candidatos(data, posicao, atributo, porcentagem):
    if posicao == "top":
        ascending = False
    elif posicao == "bottom":
        ascending = True

    data_return = data.sort_values(by=atributo, ascending=ascending)
    data_return = data_return.iloc[:int(data_return.shape[0] * porcentagem)]
    return data_return

def get_medias_br():
    """
    Retorna as médias brasileiras dos atributos do Mindmatch e seus desvios padrões
    :return: Data frame com as médias e desvios padrões dos atributos do Mindmatch
    """
    import pandas as pd
    return pd.DataFrame({'Perfil': {0: 'perfil-Capacidade analítica',
                                    1: 'perfil-Pensamento conceitual',
                                    2: 'perfil-Reflexão',
                                    3: 'perfil-Pensamento criativo',
                                    4: 'perfil-Planejamento e organização',
                                    5: 'perfil-Comunicação',
                                    6: 'perfil-Consideração pelos outros',
                                    7: 'perfil-Influência',
                                    8: 'perfil-Sociabilidade',
                                    9: 'perfil-Facilitação',
                                    10: 'perfil-Flexibilidade',
                                    11: 'perfil-Estabilidade emocional',
                                    12: 'perfil-Ambição',
                                    13: 'perfil-Iniciativa',
                                    14: 'perfil-Assertividade',
                                    15: 'perfil-Tomada de riscos',
                                    16: 'Motivacional',
                                    17: 'Social',
                                    18: 'Raciocínio',
                                    19: 'Adaptativo'},
                         'media': {0: 55.04,
                                   1: 45.32,
                                   2: 39.4,
                                   3: 44.01,
                                   4: 55.8,
                                   5: 53.76,
                                   6: 49.08,
                                   7: 49.86,
                                   8: 55.37,
                                   9: 51.18,
                                   10: 47.52,
                                   11: 54.86,
                                   12: 46.93,
                                   13: 49.42,
                                   14: 40.52,
                                   15: 40.11,
                                   16: 55.42,
                                   17:44.49,
                                   18: 36.98,
                                   19: 50.56},
                         'desvio': {0: 17.96,
                                    1: 17.77,
                                    2: 17.22,
                                    3: 18.77,
                                    4: 18.41,
                                    5: 20.41,
                                    6: 18.05,
                                    7: 19.23,
                                    8: 22.21,
                                    9: 19.36,
                                    10: 15.8,
                                    11: 20.99,
                                    12: 19.36,
                                    13: 16.73,
                                    14: 15.97,
                                    15: 16.44,
                                    16: 17.53,
                                    17: 19.6,
                                    18: 17.17,
                                    19: 18.42}})


        