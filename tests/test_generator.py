from nexus.generator import generate_data

def test_generator_reproducible(tmp_path):
    a=generate_data(1000,7,str(tmp_path/"a"))["transactions"]
    b=generate_data(1000,7,str(tmp_path/"b"))["transactions"]
    assert a.equals(b)
    assert len(a)==1000
    assert a.is_fraud.isin([0,1]).all()
