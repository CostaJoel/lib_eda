def remover_outliers_IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print("IQR:", IQR, sep="\n")
    print("Tamanho do modelo antes:", df.shape)

    df_sem_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    print("Tamanho do modelo depois:", df_sem_outliers.shape)

    return df_sem_outliers


