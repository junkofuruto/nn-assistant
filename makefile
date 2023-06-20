build: ./src/lib.rs
	wasm-pack build --target web
	cp ./pkg/*.js ./deploy/script
	cp ./pkg/*.wasm ./deploy/script

deploy:
	python -m http.server -d deploy