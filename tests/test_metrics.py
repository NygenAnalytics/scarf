import numpy as np


def test_metric_lisi(datastore, make_graph):
    lables = np.random.randint(0, 2, datastore.cells.N)
    datastore.cells.insert(
        column_name="samples_id",
        values=lables,
        overwrite=True,
    )
    lisi = datastore.metric_lisi(
        label_colnames=["samples_id"], save_result=False, return_lisi=True
    )
    assert len(lisi[0][1]) == len(datastore.cells.active_index("I"))


def test_metric_silhouette(datastore, make_graph, leiden_clustering):
    _ = datastore.metric_silhouette()


def test_metric_integration_ari(datastore, make_graph, leiden_clustering):
    lables1 = np.random.randint(0, 2, datastore.cells.N)
    lables2 = np.random.randint(0, 2, datastore.cells.N)
    datastore.cells.insert(
        column_name="lables1",
        values=lables1,
        overwrite=True,
    )
    datastore.cells.insert(
        column_name="lables2",
        values=lables2,
        overwrite=True,
    )
    _ = datastore.metric_integration(batch_labels=["lables1", "lables2"], metric="ari")

def test_metric_integration_nmi(datastore, make_graph, leiden_clustering):
    lables1 = np.random.randint(0, 2, datastore.cells.N)
    lables2 = np.random.randint(0, 2, datastore.cells.N)
    datastore.cells.insert(
        column_name="lables1",
        values=lables1,
        overwrite=True,
    )
    datastore.cells.insert(
        column_name="lables2",
        values=lables2,
        overwrite=True,
    )
    _ = datastore.metric_integration(batch_labels=["lables1", "lables2"], metric="nmi")