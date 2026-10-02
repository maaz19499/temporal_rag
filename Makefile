.PHONY: test eval demo cli lint clean

test:
	pytest tests/

eval:
	python evaluation.py

demo:
	streamlit run demo_app.py

cli:
	python cli_demo.py

lint:
	python -m py_compile evaluation.py ingestion.py retriever.py temporal_parser.py demo_app.py

clean:
	rm -rf __pycache__ .pytest_cache results/*.json results/*.md
