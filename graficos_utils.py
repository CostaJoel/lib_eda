import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

import datasets_utils

def heatmap_correlacao_empilhada(dataset,
                                 atributo_corr,
                                 atributos_remover=None,
                                 titulo_grafico=None,
                                 kind="pearson",
                                 savefig=False,
                                 nome_fig="heatmap_correlacao_empilhada",
                                 caminho_fig=None,
                                 figsize=(5, 10)):
    """
    Plota o mapa de calor representando a correlação entre os atributos em dataset com o
     atributo selecionado (atributo_corr)
    :param dataset:
    :param atributo_corr:
    :param atributos_remover:
    :param titulo_grafico:
    :param kind:
    :param savefig:
    :param nome_fig:
    :param caminho_fig:
    :param figsize:
    :return:
    """

    correlations = dataset.corr(method=kind)[atributo_corr][:]  # Substituir o dataset
    if atributos_remover is None:
        atributos_remover = atributo_corr
    crr = pd.DataFrame(correlations.drop(atributos_remover))
    crr = crr.sort_values(atributo_corr, ascending=False)
    plt.clf()

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_context("notebook")
    sns.heatmap(crr, annot=True, center=0, cmap="coolwarm", linewidths=0.2, annot_kws={"fontsize":14}, ax=ax)
    ax.autoscale(tight=True)
    ax.set_ylabel(ylabel=None)

    if titulo_grafico is None:
        ax.set_title(atributo_corr + "-" + kind)
    else:
        ax.set_title(titulo_grafico)

    if caminho_fig is not None:
        if not os.path.isdir(caminho_fig):
            os.mkdir(caminho_fig)
        nome_fig = os.path.join(caminho_fig, nome_fig)

    fig.savefig(nome_fig, dpi=300, bbox_inches="tight")
    plt.show()


def heatmap_correlacao(dataset,
                       savefig=False,
                       nome_figure="heatmap_correlacao",
                       caminho_salvar_fig="",
                       titulo_grafico=None,
                       figsize=(16, 12)):
    """
    Plota o mapa de calor com as correlações entre todos atributos do dataset.
    :param dataset:
    :param savefig:
    :param nome_figure:
    :param caminho_salvar_fig:
    :param titulo_grafico:
    :return:
    """

    # ---- Correlacao em matriz heatmapfig = plt.figure(figsize=(8,6))
    correlation_matrix = dataset.corr()
    heatmapfig = plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, vmax=0.8, square=True, linewidths=.05, annot=True)
    if savefig:
        if titulo_grafico is None:
            path = os.path.join(caminho_salvar_fig + ".png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            path = os.path.join(caminho_salvar_fig, titulo_grafico + ".png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def boxplot_agrupados_com_media(dataset,
                                atributo_classe,
                                atributos_extras=None,
                                savefig=False,
                                titulo_grafico="Performadores",
                                name_fig="boxplot_agrupados_com_medias",
                                caminho_fig=None):
    """
    Plota boxplots de cada atributo por classe (atributo_classe) com marcações
    nos gráficos que representam a média e o desvio padrão (meio desvio para baixo e para cima) brasileiro.

    :param dataset: data frame com os dados
    :param atributo_classe: atributo classe para comparação. E.g. performa {0, 1}
    :param atributos_extras: outros atributos que serão plotados juntos, mas sem a média
    :param savefig: salvar a figura ou não
    :param titulo_grafico:
    :param name_fig: nome do arquivo final
    :param caminho_fig: caminho para salvar a figura
    :return:
    """

    medias_br = datasets_utils.get_medias_br().set_index("Perfil")

    dsGrafico = dataset
    # sns.reset_orig()
    fig = plt.figure(figsize=(25, 15))
    perfil_colunas = []
    for i in dsGrafico.columns:
        if "perfil" in i:
            perfil_colunas.append(i)

    j = 0
    if atributos_extras is not None:
        for i in atributos_extras:
            plt.subplot(8, 4, j + 1)
            j += 1
            ax = sns.boxplot(data=dsGrafico, x=atributo_classe, y=i)
            ax.set_ylim((0, 110))
            ax.set_xlabel(atributo_classe)  # , fontsize=14)
            ax.set_ylabel(i)  # , fontsize=14)
            ax.figure.set_size_inches((10, 20))

    for i in perfil_colunas:
        plt.subplot(8, 4, j + 1)
        j += 1
        ax = sns.boxplot(data=dsGrafico, x=atributo_classe, y=i)
        ax.set_ylim((0, 110))
        ax.set_xlabel(atributo_classe)  # , fontsize=14)
        ax.set_ylabel(i)  # , fontsize=14)
        ax.figure.set_size_inches((10, 20))

        ax.hlines(
            y=medias_br.loc[i, "media"],
            xmin=ax.get_xlim()[0],
            xmax=ax.get_xlim()[1],
            colors='red',
            linestyles='solid')

        ax.hlines(
            y=medias_br.loc[i, "media"] + (medias_br.loc[i, "desvio"] / 2),
            xmin=ax.get_xlim()[0],
            xmax=ax.get_xlim()[1],
            colors='black',
            linestyles='dashed'
        )

        ax.hlines(
            y=medias_br.loc[i, "media"] - (medias_br.loc[i, "desvio"] / 2),
            xmin=ax.get_xlim()[0],
            xmax=ax.get_xlim()[1],
            colors='black',
            linestyles='dashed'
        )

        fig.suptitle("Performadores " + titulo_grafico)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        if savefig:
            if caminho_fig is None:
                plt.savefig(name_fig, dpi=300, bbox_inches='tight')
            else:
                if not os.path.isdir(caminho_fig):
                    os.mkdir(caminho_fig)
                path = os.path.join(caminho_fig, name_fig + ".png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


def cdf_agrupados(dataset,
                  medias_br,
                  atributo_classe,
                  atributos_extras=None,
                  savefig=False,
                  titulo_grafico="Performadores",
                  name_fig="cdf_agrupados",
                  caminho_fig=None):
    """
    Plota boxplots de cada atributo por classe (atributo_classe) com marcações
    nos gráficos que representam a média e o desvio padrão (meio desvio para baixo e para cima) brasileiro.

    :param dataset: data frame com os dados
    :param medias_br: data frame das informações com médias e desvio padrão
    :param atributo_classe: atributo classe para comparação. E.g. performa {0, 1}
    :param atributos_extras: outros atributos que serão plotados juntos, mas sem a média
    :param savefig: salvar a figura ou não
    :param titulo_grafico:
    :param name_fig: nome do arquivo final
    :param caminho_fig: caminho para salvar a figura
    :return:
    """

    medias_br = medias_br.set_index("Perfil")

    dsGrafico = dataset
    # sns.reset_orig()
    fig = plt.figure(figsize=(20, 10))
    plt.rcParams.update({
        "lines.color": "black",
        "patch.edgecolor": "black",
        "text.color": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "lightgray",
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white"})

    perfil_colunas = []
    for i in dsGrafico.columns:
        if "perfil" in i:
            perfil_colunas.append(i)

    j = 0
    if atributos_extras is not None:
        for i in atributos_extras:
            plt.subplot(7, 4, j + 1)
            j += 1
            ax = sns.ecdfplot(data=dsGrafico, x=dsGrafico[i], hue=atributo_classe)
            ax.set_ylim((0, 1))
            ax.set_xlabel(atributo_classe)  # , fontsize=14)
            ax.set_ylabel(i)  # , fontsize=14)
            ax.figure.set_size_inches((30, 30))

    for i in perfil_colunas:
        plt.subplot(7, 4, j + 1)
        j += 1
        ax = sns.ecdfplot(data=dsGrafico, x=dsGrafico[i], hue=atributo_classe)
        ax.set_ylim((0, 1))
        ax.set_xlabel(atributo_classe)  # , fontsize=14)
        ax.set_ylabel(i)  # , fontsize=14)
        ax.figure.set_size_inches((30, 30))

        ax.vlines(
            x=medias_br.loc[i, "media"],
            ymin=0,
            ymax=1,
            colors='red',
            linestyles='solid')

        ax.vlines(
            x=medias_br.loc[i, "media"] + (medias_br.loc[i, "desvio"] / 2),
            ymin=0,
            ymax=1,
            colors='black',
            linestyles='dashed'
        )

        ax.vlines(
            x=medias_br.loc[i, "media"] - (medias_br.loc[i, "desvio"] / 2),
            ymin=0,
            ymax=1,
            colors='black',
            linestyles='dashed'
        )

        fig.suptitle("Performadores " + titulo_grafico)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        if savefig:
            if caminho_fig is None:
                plt.savefig(name_fig, dpi=300, bbox_inches='tight')
            else:
                if not os.path.isdir(caminho_fig):
                    os.mkdir(caminho_fig)
                path = os.path.join(caminho_fig, name_fig + ".png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


def plotar_histplot_agrupados(data,
                              colunas_remover,
                              atributo_classe="classe",
                              savefig=False,
                              nome_figure="histplot_perfoma_nao_performa",
                              figsize=(20, 25)):
    """
    Plota o histograma de todos os pares de atributos em data
    :param data:
    :param colunas_remover:
    :param atributo_classe:
    :param savefig:
    :param nome_figure:
    :param figsize:
    :return:
    """

    # .loc[dsJoinTeste['perfil-Ambição']>60,
    # dsGrafico = concatenado_droped[cols+['performa2']].dropna()
    dsGrafico = data.drop(colunas_remover, axis=1)
    # dsJoinDrop2 = dsJoinDrop2[colsTeste]
    sns.reset_orig()
    fig = plt.figure(figsize=figsize)
    # for k in ['Vendedor I','Vendedor II','Vendedor III','Supervisor de Vendas']:
    j = 0
    # dsGrafico = dsJoin.loc[dsJoin['cargo']==j,colsGrafico].dropna()
    for i in dsGrafico.columns:
        plt.subplot(6, 4, j + 1)
        j += 1
        print(i)
        kde = True
        # if i =='Raciocínio' :#or i=='experiencia' or i=='HPC2020':
        #     kde=False
        sns.histplot(dsGrafico[i].loc[dsGrafico[atributo_classe] == 1], color='green', label='Performador', kde=kde)
        sns.histplot(dsGrafico[i].loc[(dsGrafico[atributo_classe] == 0)], color='red', label='Nao Performador', kde=kde)
        plt.legend(loc='best')
        fig.suptitle('Performadores')
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
    if savefig:
        plt.savefig(nome_figure)
    plt.show()


def plotar_boxplot_agrupados(data,
                             colunas_remover,
                             atributo_classe="classe",
                             savefig=False,
                             nome_figure="boxplot_perfoma_nao_performa",
                             figsize=(20, 25)):
    """
    Plota os boxplots de todos os atributos em data comparando cada um por classe (atributo_classe)
    :param data:
    :param colunas_remover:
    :param atributo_classe:
    :param savefig:
    :param nome_figure:
    :param figsize:
    :return:
    """
    # .loc[dsJoinTeste['perfil-Ambição']>60,
    # dsGrafico = concatenado_droped[cols+['performa2']].dropna()
    dsGrafico = data.drop(colunas_remover, axis=1)
    # dsJoinDrop2 = dsJoinDrop2[colsTeste]
    sns.reset_orig()
    sns.set(font_scale=1.4)
    fig = plt.figure(figsize=figsize)
    # for k in ['Vendedor I','Vendedor II','Vendedor III','Supervisor de Vendas']:
    j = 0
    # dsGrafico = dsJoin.loc[dsJoin['cargo']==j,colsGrafico].dropna()
    for i in dsGrafico.columns[:-1]:
        plt.subplot(6, 4, j + 1)
        j += 1
        # if i =='Raciocínio' :#or i=='experiencia' or i=='HPC2020':
        #     kde=False
        ax = sns.boxplot(data=dsGrafico, x='classe', y=i)
        ax.set_ylim((0, 100))
        ax.set_xlabel('classe', fontsize=14)
        ax.set_ylabel(i, fontsize=14)
        fig.suptitle('Performadores')
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
    if savefig:
        plt.savefig(nome_figure)
    plt.show()


def compara_medias_grafico_linhas(dados_mapeamento,
                                  nome_cliente="mapeamento",
                                  figsize=(15, 3)):
    medias_mapeamento = dados_mapeamento.describe().loc['mean', :]
    medias_mapeamento = medias_mapeamento.reset_index()
    medias_mapeamento.columns = ['Perfil', 'media']
    medias_br = datasets_utils.get_medias_br()
    aux = medias_br.iloc[:, :2]
    aux['classe'] = 'BR'
    aux = pd.concat([aux, medias_mapeamento])
    aux.fillna(nome_cliente, inplace=True)

    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=aux, x='Perfil', y='media', hue='classe', style='classe', markers=True, dashes=False)
    ax.set_ylim([0, 100])
    plt.xticks(rotation=70)
    plt.show()


def plotar_distribuicoes_bivariadas(data,
                                    rotulo,
                                    atributo,
                                    kde=True,
                                    colors={1: 'darkblue', 0: 'red'},
                                    stat='density',
                                    element='step',
                                    labels={1: 'Performa', 0: 'Não performa'},
                                    fig_size_inches = (12,5),
                                    save_fig = False,
                                    nome_figura = None,
                                    caminho_figura = None
                                    ):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2)

    sns.histplot(data.loc[data[rotulo] == 1, atributo],
                 kde=kde,
                 color=colors.get(1),
                 stat=stat,
                 element=element,
                 label=labels.get(1),
                 ax=axes[0])

    sns.histplot(data.loc[data[rotulo] == 0, atributo],
                 kde=kde,
                 color=colors.get(0),
                 stat=stat,
                 element=element,
                 label=labels.get(0),
                 ax=axes[0])

    axes[0].figure.set_size_inches(fig_size_inches)
    axes[0].set_title('Histograma')
    if stat == 'density':
        label = 'Densidade'
    elif stat == 'probability':
        label = 'Probabilidade'
    else:
        label = 'Cont'
    axes[0].set_ylabel(label)
    # axes[0].set_xlim(0, 100)
    axes[0].legend()

    sns.ecdfplot(data.loc[data[rotulo] == 1, atributo],
                 label=labels.get(1),
                 color=colors.get(1),
                 ax=axes[1])

    sns.ecdfplot(data.loc[data[rotulo] == 0, atributo],
                 label=labels.get(0),
                 color=colors.get(0),
                 ax=axes[1])

    axes[1].figure.set_size_inches(fig_size_inches)
    axes[1].set_title('ECDF')
    axes[1].set_ylabel('Proporção')
    # axes[1].set_xlim(0, 100)
    axes[1].legend()

    if save_fig:
        if nome_figura is not None:
            nome_figura = "distribuicao_" + atributo
        save_fig(nome_figura=nome_figura, caminho_figura=caminho_figura)

    plt.show()


def save_fig(nome_figura, caminho_figura=None):
    if caminho_figura is None:
        plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
    else:
        if not os.path.isdir(caminho_figura):
            os.mkdir(caminho_figura)
        path = os.path.join(caminho_figura, nome_figura + ".png")
        plt.savefig(path, dpi=300, bbox_inches='tight')


def lineplot_comparacao_media_br(df,
                                 medias_br,
                                 nome_atributo_rotulo = 'performa',
                                 plotar_low_perf=True,
                                 nome_atributo_low_perf = 'low_perf'):

    from mindsight import datasets_utils
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.io as pio


    pio.renderers.default = 'iframe'  # or 'notebook' or 'colab'

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(y=df.loc[df[nome_atributo_rotulo] == 1, list(medias_br.Perfil.values)].mean(),
                   x=list(medias_br.Perfil.values),
                   name="Performadores",
                   line=dict(color='#166889', width=2, dash='solid')
                   ))

    fig.add_trace(go.Scatter(y=medias_br.loc[list(medias_br.Perfil.values), 'media'],
                             x=list(medias_br.Perfil.values),
                             name='População',
                             line=dict(color='#919191', width=2, dash='solid')
                             ))

    fig.add_trace(
        go.Scatter(y=df.loc[df[nome_atributo_rotulo] == 0, list(medias_br.Perfil.values)].mean(),
                   x=list(medias_br.Perfil.values),
                   name="Não Performadores",
                   line=dict(color='green', width=2, dash='solid')
                   ))

    if plotar_low_perf:
        fig.add_trace(
            go.Scatter(y=df.loc[df[nome_atributo_low_perf] == 1, list(medias_br.Perfil.values)].mean(),
                       x=list(medias_br.Perfil.values),
                       name="Low Perfs",
                       line=dict(color='red', width=2, dash='solid')
                       ))

    fig.update_layout(title='<b> Performadores x Médias BR<b>',
                      xaxis_title='Atributos',
                      yaxis_title='Valor Médio',
                      titlefont={'size': 28, 'family': 'Serif'},
                      template='simple_white',
                      showlegend=True,
                      width=900, height=500)

    fig.update_yaxes(title_text="<b>valores</b>",
                     #                  secondary_y=True,
                     range=[0, 100])

    fig.show()


def plot_atributos_diferenca_media(df, cols_medias, raciocinio = "adaptativo", font_scale=1.5,nome_atributo_rotulo = 'performa', save_fig = False, p_min = 0.05):
    from scipy.stats import ttest_1samp
    from mindsight import datasets_utils

    medias_br = datasets_utils.get_medias_br()

    if raciocinio.lower() == 'adaptativo':
        medias_br = medias_br.set_index('Perfil').drop('Raciocínio').rename(
            index={'Adaptativo': 'Adaptivo Transformado'})
    elif raciocinio.lower() == 'fluido':
        medias_br = medias_br.set_index('Perfil').drop('Adaptativo')
    else:
        print('Testes Operacionais')
        medias_br = medias_br.set_index('Perfil').drop(['Adaptativo', 'Raciocínio'])
        try:
            cols_medias.remove('Raciocínio')
        except:
            print('Não possui Raciocínio Fluído')

        try:
            cols_medias.remove('Adaptivo Transformado')
        except:
            print('Não possui Adaptivo Transformado')

    try:
        cols_medias.remove('Cultura pontuação')
    except:
        print('Colunas sem Cultura')

    caracateristicas_diferem_populacao = dict()
    for i in cols_medias:
        # print(df.loc[df[nome_atributo_rotulo] == 1, i])
        tset, pval = ttest_1samp(df.loc[df[nome_atributo_rotulo] == 1, i],
                                 medias_br.loc[i, 'media'])
        if pval <= p_min:
            caracateristicas_diferem_populacao[i] = {'t_score': round(tset, 2),
                                                     'p-valor': round(pval, 3)}

    diferenca_populacao = pd.DataFrame(caracateristicas_diferem_populacao).T

    try:
        diferenca_populacao['label'] = diferenca_populacao['t_score'].apply(lambda x: 1 if x <= 0 else 0)
    except KeyError:
        print('Não existe nenhum atributo com significativa diferença da média!!')
        return

    diferenca_populacao['diferença'] = [
        df.loc[df[nome_atributo_rotulo] == 1, x].mean() - medias_br.loc[x, 'media'] for
        x in list(diferenca_populacao.index)]
    diferenca_populacao = diferenca_populacao[diferenca_populacao['diferença'].abs() > 4]
    # diferenca_populacao.drop('Adaptivo Transformado', inplace=True)
    diferenca_populacao = diferenca_populacao.round(2)
    print(diferenca_populacao)

    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.figure()

    ax = sns.barplot(data=diferenca_populacao.sort_values(by="diferença", ascending=False),
                     x='diferença',
                     y=diferenca_populacao.sort_values(by="diferença", ascending=False).index,
                     hue="label",
                     palette={1: '#919191',
                              0: '#166889'})

    for i, index in enumerate(diferenca_populacao.sort_values(by="diferença", ascending=False).index):
        ax.text(
            x=diferenca_populacao.loc[index, 'diferença'] - 1 if diferenca_populacao.loc[index, 'diferença'] > 0 else
            diferenca_populacao.loc[index, 'diferença'] + 1.2,
            y=i + 0.5 if diferenca_populacao.loc[index, 'diferença'] < 0 else i - 0.1,
            s=diferenca_populacao.loc[index, 'diferença'], color='black', ha='center')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.figure.set_size_inches(12, 10)
    ax.legend("")
    # ax.set_title('Atributos médias significativamente diferentes da população', fontsize=14, fontweight='bold')
    if save_fig:
        plt.savefig('diferenca_performadores_media.png', dpi=300, bbox_inches="tight")

    plt.show()

