import platform
import sys
import json
import cobra
import cplex
import re
import os
from os.path import exists
import logging
from configparser import ConfigParser
config = ConfigParser()
config.read("config.cfg")
paths = config.get("script", "syspaths").split(";")
for path in paths:
    sys.path.append(path)
import cobrakbase
from escher import Builder
from optlang.symbolics import Zero, add
from modelseedpy import MSPackageManager, MSGapfill, MSGrowthPhenotypes, MSModelUtil, MSATPCorrection
from cobrakbase.core.kbasefba.newmodeltemplate_builder import NewModelTemplateBuilder
from annotation_ontology_api.annotation_ontology_apiServiceClient import annotation_ontology_api
from modelseedpy.helpers import get_template
import pandas as pd

kbase_api = cobrakbase.KBaseAPI()
anno_api = annotation_ontology_api()
glc_o2_atp_media = kbase_api.get_from_ws("Glc.O2.atp",94026)
gmm = kbase_api.get_from_ws("Carbon-D-Glucose","KBaseMedia")
template = kbase_api.get_from_ws("GramNegModelTemplateV4","NewKBaseModelTemplates")
core = NewModelTemplateBuilder.from_dict(get_template('template_core'), None).build()
types = ["RAST","DRAM","DRAM.RAST"]
genomeid = sys.argv[1]
genomews = int(sys.argv[2])
modelws = int(sys.argv[3])
#Computing reaction scores
reaction_genes = {}
output = anno_api.get_annotation_ontology_events({
    "input_ref" : str(genomews)+"/"+genomeid,
})
events = output["events"]
for event in events:
    for gene in event["ontology_terms"]:
        for term in event["ontology_terms"][gene]:
            if "modelseed_ids" in term:
                for rxn in term["modelseed_ids"]:
                    newrxn = re.sub("^MSRXN:","",rxn)
                    if newrxn not in reaction_genes:
                        reaction_genes[newrxn] = {}
                    if gene not in reaction_genes[newrxn]:
                        reaction_genes[newrxn][gene] = 0            
                    reaction_genes[newrxn][gene] += 1
#Gapfilling RAST, DRAM, and RAST.DRAM models
for mdltype in types:
	mdlid = genomeid+"."+mdltype+".mdl"
	if not exists("GFModels/"+mdltype+"/"+genomeid+".json"):
		#Getting model, media, templates
		model = kbase_api.get_from_ws(mdlid,modelws)
		model.solver = 'optlang-cplex'
		#Computing ATP to build tests to ensure gapfilling doesn't break ATP
		atpmethod = MSATPCorrection(model,core,[glc_o2_atp_media])
		evaluation = atpmethod.evaluate_growth_media()
		atpmethod.restore_noncore_reactions()
		threshold = 4*evaluation[glc_o2_atp_media.id]
		if threshold > 50:
			threshold = 50
		tests = [{"media":glc_o2_atp_media,"objective":atpmethod.atp_hydrolysis.id,"is_max_threshold":True,"threshold":threshold}]
		#Gapfilling
		msgapfill = MSGapfill(model,[template],[],tests,reaction_genes,[])
		gfresults = msgapfill.run_gapfilling(gmm,"bio1")
		if gfresults == None:
			print("Gapfilling failed on "+mdlid)
		else:
			model = msgapfill.integrate_gapfill_solution(gfresults)
			pkgmgr = MSPackageManager.get_pkg_mgr(model)
			pkgmgr.getpkg("KBaseMediaPkg").build_package(gmm)
			solution = model.optimize()
			print(mdltype+"\t"+genomeid+"\t"+str(solution.objective_value))
			if solution.objective_value > 0.01:
				cobra.io.save_json_model(model,"GFModels/"+mdltype+"/"+genomeid+".json")
			else:
				print("Model failed validation and will not be saved!")