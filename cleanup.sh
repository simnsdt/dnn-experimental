#!/bin/bash

DELETE_MODELS="false"
DELETE_RESULTS="true"
DELETE_CACHE="true"
# Delete results:
if [ "$DELETE_RESULTS" = true ]; then
	find . -iname 'results*' -type f | xargs rm -f
fi
# Delete  models
if [ "$DELETE_MODELS" = true ]; then
	find . -iname 'VGG19*' -type f | xargs rm -f
	find . -iname 'ResNet50*' -type f | xargs rm -f
fi

# Delete python cache:
if [ "$DELETE_CACHE" = true ]; then
	find . -iname '*pycache*' | xargs rm -r -f
	find . -iname '*.pyc' | xargs rm -f
fi
echo "CLEANUP DONE!"
