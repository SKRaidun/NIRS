void processing() {

    TH1F *hist = new TH1F("hist", "histogram", 170, 35, 52);
    // Считывание исходника
    fstream file;
    file.open("/Users/semenraydun/Desktop/NIRS/RUN010/RUN010_6He_bin/RUN010_dE2(6He)_20.7.txt", ios::in);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        float energ;
        float count;
        ss>>energ>>count;
        hist->Fill(energ, count);
    }
    file.close();

    
//Считывание питоновского файла
    TTree *t = new TTree("t", "tree from 10_dE2");
    t->ReadFile("/Users/semenraydun/Documents/GitHub/NIRS/10_dE2.csv", "Ядро/C:Уровень/F:Xc/F:Xcmin/F:Xcmax/F:Wmin/F:Wmax/F:Amin/F:Amax/F");
  
    char nucleus[3];
    Float_t xc, xmin, xmax, Wmin, Wmax, Amin, Amax, lvl, Xtotmin = 35, Xtotmax = 52, Ampl,dAmpl,Summ,dSumm;
   t->SetBranchAddress("Ядро", &nucleus);
   t->SetBranchAddress("Уровень", &lvl);
   t->SetBranchAddress("Xc", &xc);
   t->SetBranchAddress("Xcmin", &xmin);
   t->SetBranchAddress("Xcmax", &xmax);
   t->SetBranchAddress("Wmin", &Wmin);
   t->SetBranchAddress("Wmax", &Wmax);
   t->SetBranchAddress("Amin", &Amin);
   t->SetBranchAddress("Amax", &Amax);
   
    TCanvas *c1 = new TCanvas();
    hist->SetMarkerStyle(3);
    hist->Draw();
    //Дальше все для total
    string gauss1;
    // Формирую строку по примеру gausn(0)+gausn(3)+...
    for (int i = 0; i < 45; i++) {
        if (i == 44) {
            gauss1 += "gausn("+to_string(3*i)+")"; 
        } else {
        gauss1 += "gausn("+to_string(3*i)+")"+"+";
        }
    }
    char const *gauss = gauss1.c_str();

    TF1 *total = new TF1("total",gauss,Xtotmin,Xtotmax);
    total->SetLineColor(5);

    Double_t par[132];
// 
   for (int i = 0; i < 45; i++) {
   t->GetEntry(i);
   total->SetParLimits(3*i,Amin,10000);
   par[3*i+1]=xc; total->FixParameter(3*i+1,par[3*i+1]);
   par[3*i+2]=Wmin+0.1; total->FixParameter(3*i+2,par[3*i+2]);
    }

    hist->Fit(total);
    ofstream fout("Fit10_dE2.txt");
    fout<<"E_x"<<"        "<<"Summ"<<"        "<< "d_Summ"<<endl; 
    total->GetParameters(par);
    // Формирую TF для каждого уровня
        for (int i = 0; i < 45; i++) {
        t->GetEntry(i);
        TF1 *g = new TF1("g", "gausn", Xtotmin, Xtotmax);
         if (i <= 10) {
             g->SetLineColor(2);
        } else if (i <= 22) {
            g->SetLineColor(3);
        } else if (i <= 38) {
            g->SetLineColor(1);
        } else if (i <= 43) {
            g->SetLineColor(9);
        }
        g->SetParameters(&par[3*i]);
        g->Draw("Same");

    Ampl= total->GetParameter(3*i);
    dAmpl= total->GetParError(3*i);
    Summ=10*Ampl;	
    dSumm=10*dAmpl;	
    fout<<lvl<<"        "<<Summ<<"        "<< dSumm<<endl;; 	
    }
    fout.close();
}
