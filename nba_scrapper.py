def scrape_totals(year):
    """
    NBA totals data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    #out_path = NBAanalysisdir + 'data/NBA_totals_{}-{}.csv'.format(year-1, year)
    out_path = 'data/NBA_totals_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', \
                '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}_totals.html'.format(year)

    print("--- Scraping totals data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="all_totals_stats")
    cells = table.find_all('td')

    ncolumns = len(features)
    
    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Pos     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    GS      = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    FG      = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    FGA     = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    FGP     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    THP     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    THPA    = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    THPP    = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    TWP     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    TWPA    = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    TWPP    = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    EFGP    = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    FT      = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    FTA     = [cells[i].getText() for i in range(18, len(cells), ncolumns)]
    FTP     = [cells[i].getText() for i in range(19, len(cells), ncolumns)]
    ORB     = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    DRB     = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    TRB     = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    AST     = [cells[i].getText() for i in range(23, len(cells), ncolumns)]
    STL     = [cells[i].getText() for i in range(24, len(cells), ncolumns)]
    BLK     = [cells[i].getText() for i in range(25, len(cells), ncolumns)]
    TOV     = [cells[i].getText() for i in range(26, len(cells), ncolumns)]
    PF      = [cells[i].getText() for i in range(27, len(cells), ncolumns)]
    PTS     = [cells[i].getText() for i in range(28, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Pos[i], Age[i], Tm[i], G[i], GS[i], MP[i], FG[i], FGA[i], FGP[i], THP[i], THPA[i], THPP[i], TWP[i], TWPA[i], \
               TWPP[i], EFGP[i], FT[i], FTA[i], FTP[i], ORB[i], DRB[i], TRB[i], AST[i], STL[i], BLK[i], TOV[i], PF[i], PTS[i]]
        csv_writer.writerow(row)
        
        

def scrape_advanced(year):
    """
    NBA advanced data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = 'data/NBA_advanced_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', \
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
        
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'.format(year)

    print("--- Scraping advanced data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="all_advanced_stats")
    cells = table.find_all('td')

    ncolumns = len(features) + 2 # plus 2 because there are two columns missing!

    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Pos     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    PER     = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    TSP     = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    TPAr    = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    FTr     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    ORBP    = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    DRBP    = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    TRBP    = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    ASTP    = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    STLP    = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    BLKP    = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    TOVP    = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    USGP    = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    OWS     = [cells[i].getText() for i in range(19, len(cells), ncolumns)] # 18 is empty!
    DWS     = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    WS      = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    WS48    = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    OBPM    = [cells[i].getText() for i in range(24, len(cells), ncolumns)] # 23 is empty!
    DBPM    = [cells[i].getText() for i in range(25, len(cells), ncolumns)]
    BPM     = [cells[i].getText() for i in range(26, len(cells), ncolumns)]
    VORP    = [cells[i].getText() for i in range(27, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Pos[i], Age[i], Tm[i], G[i], MP[i], PER[i], TSP[i], TPAr[i], FTr[i], ORBP[i], DRBP[i], TRBP[i], \
               ASTP[i], STLP[i], BLKP[i], TOVP[i], USGP[i], OWS[i], DWS[i], WS[i], WS48[i], OBPM[i], DBPM[i], BPM[i], VORP[i]]
        csv_writer.writerow(row)

    return 0