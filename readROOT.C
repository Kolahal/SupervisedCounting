void readROOT()
{
	TFile* f = new TFile("lAr_neutron_test.root", "read");
	//TFile* f = new TFile("/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/Alpha/alpha_decay_type11/output_alpha_decay_type11_18.root","read");
	TTree* t1;
	f->GetObject("T", t1);
	
	RAT::DS::Root *ds;
	RAT::DSReader dsReader("lAr_neutron_test.root");
	//RAT::DSReader dsReader("/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/Alpha/alpha_decay_type11/output_alpha_decay_type11_18.root");
	
	t1->SetBranchAddress("ds", &ds);
	
	float mcnPE = 0.0;
	float recPE = 0.0;
	
	while (ds = dsReader.NextEvent())
	{
		RAT::DS::MC* mc = ds->GetMC();
		for(int k =0; k<ds->GetEVCount(); k++)
		{
			RAT::DS::EV* Ev = ds->GetEV(k);
			mc_nPE=mc->GetNumPE()*1.0;
			recPE =Ev->GetQPE();
			//cout<<mc_nPE<<"     "<<recPE<<endl;
			cout<<recPE<<endl;
		}
	}
	
	return;
}
