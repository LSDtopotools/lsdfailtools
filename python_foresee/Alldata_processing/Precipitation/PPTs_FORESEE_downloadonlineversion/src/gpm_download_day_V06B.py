
        for i in range(0,len(years),1):

            month = []
            months = []

            month = list(map(str,range(1,13)))

            months = list(map(lambda x: '0' + x if len(x)==1 else x,month))

            if years[i] == str_Start_Date[0]:
                try:
                    del months[:months.index(str_Start_Date[1])]
                except:
                    pass

            if years[i] == str_End_Date[0]:
                try:
                    del months[months.index(str_End_Date[1])+1:]
                except:
                    pass

            for j in range(0,len(months),1):
                #print months[j]
                url ='https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/'+years[i]+'/'+months[j]+'/'
                print (url)

                #quit()
                #Acess the URL
                try:
                    urlpath =urlopen(url)
                except:
                    continue
                #Decode the URL
                string = urlpath.read().decode('utf-8')

                #Extract HDF5 files and make an file list
                pattern = re.compile('3B.*?nc4.*?')
                filelist = list(set(list(map(str,pattern.findall(string)))))
                filelist.sort()
                #print (filelist)


                try:
                    try:
                        startImg = filelist.index('3B-DAY.MS.MRG.3IMERG.'+str_Start_Date[0]+str_Start_Date[1]+str_Start_Date[2]+'-S000000-E235959.V06.nc4')
                    except:
                        startImg = filelist.index('3B-DAY.MS.MRG.3IMERG.20000601-S000000-E235959.V06.nc4')
                except:
                    startImg = None

                #DEL UNDER START
                if years[i] == str_Start_Date[0]:
                    if months[j] == str_Start_Date[1]:
                        #Start month
                        try:
                            del filelist[:startImg]
                        except:
                            pass
                    else:
                        pass
                else:
                    pass

                try:
                    endImg = filelist.index('3B-DAY.MS.MRG.3IMERG.'+str_End_Date[0]+str_End_Date[1]+str_End_Date[2]+'-S000000-E235959.V06.nc4')

                except:
                    endImg = None

                #DEL OVER END
                if years[i] == str_End_Date[0]:
                    if months[j] == str_End_Date[1]:
                        #End month
                        try:
                            del filelist[endImg+1:]
                        except:
                            pass
                else:
                    pass

                filteredList = filelist #= list(filter(lambda x: x not in os.listdir(input_dir),filelist))

                #print(filteredList)

                for item in range(0,len(filteredList)):

                    os.system('wget --user=' + GetLoginInfo[0] + ' --password=' + GetLoginInfo[1] + ' --show-progress -c -q '+  url + filteredList[item] + ' -O ' + input_dir + backslh + filteredList[item])

    except:
        print ('\nDownloads finished')
    print ('\nDownloads finished')
