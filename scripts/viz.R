library(ggplot2)
library(viridis)
library(grid)
library(gridExtra)
library(dplyr)
library(matrixStats)
library(ggrepel)
library(tidyr)
library(png)
NUM_EXPERTS = 14
NUM_AMATEURS = 120
NUM_TOTAL = 134
COLOR_START = 0.75
COLOR_END = 0.1
k = c(1,2,3,5,7,9,11,13,17,19,23,29,50,75,100,125,NUM_EXPERTS+NUM_AMATEURS-1)
rho = c(0,0.25,0.5,0.75,1,1.25,1.5)

# Shortcut - cmd + option + t to run code chunk

#### Some prepping for visualization ####
# the parameters are 17 * 7, so choose k=5,rho=1, so choose 26
stats_df = read.table(file.path('results','stats.csv'),
                      header=TRUE,sep=',')

correlations = read.table(file.path('results','correlation.csv'),
                          header=TRUE,sep=',', row.names=1)

both_error = read.table(file.path('results','simulations',
                                  'both','performance.csv'),header=FALSE,sep=' ')
both_error = both_error[,26]
experts_error = read.table(file.path('results','simulations',
                                     'experts','performance.csv'),header=FALSE,sep=' ')
experts_error = experts_error[,26]
amateurs_error = read.table(file.path('results','simulations',
                                     'amateurs','performance.csv'),header=FALSE,sep=' ')
amateurs_error = amateurs_error[,26]
# get set of potential correlations from the corr mat for k = 5 and rho = 1
potential_mat = correlations
temp = rowMeans(potential_mat, na.rm=T)
for (r in 1:nrow(potential_mat)){
  potential_mat[r,is.na(potential_mat[r,])] = temp[r]
}
diag(potential_mat) = NA
potential_mat = as.matrix(potential_mat)
for (r in 1:nrow(potential_mat)){
  potential_row = potential_mat[r,]
  k_val = sort(potential_row, decreasing=TRUE)[6]
  potential_mat[r,] = (potential_row > k_val) * potential_row
}
# get influence as the column sum of potential_mat
potential_influence = colSums(potential_mat, na.rm=TRUE)
potential_influence = potential_influence + min(potential_influence[which(potential_influence > 0)])
potential_influence = potential_influence / sum(potential_influence, na.rm=TRUE)
# add to stats_df
stats_df$potential_influence = potential_influence
stats_df$no_influence = min(potential_influence) * 2
# also include potential influence in the crowd (k = NUM_TOTAL-1)
potential_mat = correlations
temp = rowMeans(potential_mat, na.rm=T)
for (r in 1:nrow(potential_mat)){
  potential_mat[r,is.na(potential_mat[r,])] = temp[r]
}
diag(potential_mat) = NA
potential_mat = as.matrix(potential_mat)
potential_influence = colSums(potential_mat, na.rm=TRUE)
potential_influence = potential_influence + min(potential_influence[which(potential_influence > 0)])
potential_influence = potential_influence / sum(potential_influence, na.rm=TRUE)
# add to stats_df
stats_df$crowd_influence = potential_influence



#### Homophily visualisation (individual, actual influence) ####

homophily_df = read.table(file.path('results','homophily','both',
                                    'individual.csv'),header=TRUE,sep=',')
homophily_df = subset(homophily_df, select = -c(X) )


for (r in 0:(NUM_TOTAL-1)){ 
  subtitle = 'Homophily index (Hi) = similar ties /(similar ties + dissimilar ties), \nGroup weight = Number of raters in group/Total number of raters, \nCount weight = number of ratings by group / total number of ratings'
  top_panel = homophily_df %>% filter(homophily_df$method == "Real homophily")
  top_panel = top_panel %>% filter(top_panel$rater == r)
  top_panel$group = as.character(top_panel$group)
  top_panel = top_panel %>% mutate(group = replace(group, group == 'experts', 'Critc'))
  top_panel = top_panel %>% mutate(group = replace(group, group == 'amateurs', 'Amateur'))
  top_panel$rho = as.numeric(top_panel$rho)
  
  
  top_plot = ggplot(top_panel, aes(x=kval, y=Hi)) +  
    geom_line(aes(group=rhoval,color=as.factor(rhoval))) + 
    geom_hline(aes(yintercept=baseline_2),linetype="dashed") + 
    
    scale_colour_viridis_d(option='F',name='rho',begin=0.0, end=0.75) + 
    facet_grid(cols = vars(group)) +  theme_bw() + theme(aspect.ratio = .75) + ylim(0,1)
  
  ggsave(file.path('results','visualization', 'individual-homophily',paste(r, '.png',sep='')))
}



#### Figures for main paper : figure 4 ####

homophily_df = read.table(file.path('results','homophily','both',
                                    'group.csv'),header=TRUE,sep=',')
homophily_df = subset(homophily_df, select = -c(X) )

subtitle = 'Homophily index (Hi) = similar ties /(similar ties + dissimilar ties), \nGroup weight = Number of raters in group/Total number of raters, \nCount weight = number of ratings by group / total number of ratings'
top_panel = homophily_df %>% filter(homophily_df$method == "Real homophily")
top_panel$group = as.character(top_panel$group)
top_panel = top_panel %>% mutate(group = replace(group, group == 'experts', 'Critcs'))
top_panel = top_panel %>% mutate(group = replace(group, group == 'amateurs', 'Amateurs'))
top_panel = top_panel[top_panel$kval < 30,]

top_plot = ggplot(top_panel, aes(x=kval, y=Hi)) +  
  
  geom_line(aes(group=rhoval,color=as.factor(rhoval))) + 
  geom_hline(aes(yintercept=baseline_1),linetype="dashed") +
  geom_text(aes(25,baseline_1,label = 'group weight', vjust = 3), size=3.0, check_overlap=TRUE) + 
  
  geom_hline(aes(yintercept=baseline_2),linetype="dashed", color="blue") +
  geom_text(aes(25,baseline_2,label = 'count weight', vjust = 3), size=3.0, check_overlap=TRUE) + 
  scale_colour_viridis_d(option='F',name='rho',begin=0.0, end=0.75) + #ggtitle('k vs Homophily', subtitle = subtitle) + 
  ylim(0,1) + labs(x='Number of neighbours (k)', y='Homophily index') + 
  facet_grid(cols = vars(group)) +  theme_bw() + theme(aspect.ratio = .75, 
                                                       legend.position=c(.04,.825)) +
  scale_x_continuous(breaks=c(1,5,10,15,20,25,29))


addSmallLegend <- function(myPlot, pointSize = 1.25, textSize = 7, spaceLegend = 0.3) {
  myPlot +
    guides(shape = guide_legend(override.aes = list(size = pointSize)),
           color = guide_legend(override.aes = list(size = pointSize), reverse=TRUE)) +
    theme(legend.title = element_text(size = textSize), 
          legend.text  = element_text(size = textSize),
          legend.key.size = unit(spaceLegend, "lines"))
}

top_plot = addSmallLegend(top_plot)
top_plot
ggsave(file.path('results','visualization', 'fig_4.png'), scale=0.8)
knitr::plot_crop(file.path('results','visualization', 'fig_4.png'))

#### Homophily visualisation (group, potential influence) ####
homophily_df = read.table(file.path('results','homophily','both',
                                    'group.csv'),header=TRUE,sep=',')
homophily_df = subset(homophily_df, select = -c(X) )

subtitle = 'Homophily index (Hi) = similar ties /(similar ties + dissimilar ties), \nGroup weight = Number of raters in group/Total number of raters, \nCount weight = number of ratings by group / total number of ratings'
top_panel = homophily_df %>% filter(homophily_df$method == "Potential homophily")
top_panel$group = as.character(top_panel$group)
top_panel = top_panel %>% mutate(group = replace(group, group == 'experts', 'Critcs'))
top_panel = top_panel %>% mutate(group = replace(group, group == 'amateurs', 'Amateurs'))


top_plot = ggplot(top_panel, aes(x=kval, y=Hi)) +  
  
  geom_line(aes(group=rhoval,color=as.factor(rhoval))) + 
  geom_hline(aes(yintercept=baseline_1),linetype="dashed") +
  geom_text(aes(25,baseline_1,label = 'group weight', vjust = 1.5), size=3.0, check_overlap=TRUE) + 
  
  geom_hline(aes(yintercept=baseline_2),linetype="dashed", color="blue") +
  geom_text(aes(25,baseline_2,label = 'count weight', vjust = 1.5), size=3.0, check_overlap=TRUE) + 
  
  scale_colour_viridis_d(option='F',name='rho',begin=0.0, end=0.75) + #ggtitle('k vs Homophily', subtitle = subtitle) + 
  ylim(0,1) + labs(x='Number of neighbours (k)', y='Homophily index') + 
  facet_grid(cols = vars(group)) +  theme_bw() + theme(aspect.ratio = .75, legend.position=c(.05,.19)) +
  scale_x_continuous(breaks=c(1,25,50,75,100,125))
addSmallLegend <- function(myPlot, pointSize = 1.25, textSize = 7, spaceLegend = 0.3) {
  myPlot +
    guides(shape = guide_legend(override.aes = list(size = pointSize)),
           color = guide_legend(override.aes = list(size = pointSize), reverse=TRUE)) +
    theme(legend.title = element_text(size = textSize), 
          legend.text  = element_text(size = textSize),
          legend.key.size = unit(spaceLegend, "lines"))
}

top_plot = addSmallLegend(top_plot)
top_plot
ggsave(file.path('results','visualization', 'fig_8.png'), scale=0.8)
knitr::plot_crop(file.path('results','visualization', 'fig_8.png'))











#### Figures for main paper : figure 1 ####
ex_mc = rowMeans(as.matrix(correlations[1:NUM_EXPERTS,1:NUM_EXPERTS]),na.rm=T)
am_mc = rowMeans(as.matrix(correlations[NUM_EXPERTS+1:NUM_AMATEURS,NUM_EXPERTS+1:NUM_AMATEURS]),na.rm=T)
ex_dc = rowSds(as.matrix(correlations[1:NUM_EXPERTS,1:NUM_EXPERTS]),na.rm=T)
am_dc = rowSds(as.matrix(correlations[NUM_EXPERTS+1:NUM_AMATEURS,NUM_EXPERTS+1:NUM_AMATEURS]),na.rm=T)
stats_df$mean_corr_before = as.numeric(c(ex_mc, am_mc))
stats_df$disp_corr_before = as.numeric(c(ex_dc, am_dc))

df_before = stats_df[,c('mean_corr_before','disp_corr_before','no_influence', 'e_a')]
colnames(df_before) = c('mean_corr', 'disp_corr', 'potential_influence', 'e_a')
fig_1_df = rbind(df_before,
                 stats_df[,c('mean_corr','disp_corr','potential_influence','e_a')])
fig_1_df = cbind(fig_1_df, c( rep('Source of correlations : Group members', NUM_TOTAL), 
                              rep('Source of correlations : Everyone', NUM_TOTAL) ))
colnames(fig_1_df)[5] = 'sep_tog'
fig_1_df$e_a = rep(c(rep('Experts', NUM_EXPERTS), rep('Amateurs',NUM_AMATEURS)), 2)
expert_labels = c('WA','NM','JR',
                  'TA','B&D','JS',
                  'JL','De','RVF',
                  'JA','LeP','PW','RG','CK')
all_labels = c(expert_labels, rep('', NUM_AMATEURS))
fig_1_df$node_labels = rep(all_labels,2)

fig_1 = ggplot(data=fig_1_df, aes(x=as.numeric(mean_corr), y=as.numeric(disp_corr), size=potential_influence, 
                                  label=node_labels)) + 
  geom_point(aes(fill=e_a),shape=21, alpha=0.8) + 
  facet_grid(cols = vars(sep_tog)) +  
  facet_grid(~factor(sep_tog, levels=c('Source of correlations : Group members',
                                       'Source of correlations : Everyone'))) + 
  scale_fill_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') +
  theme_bw() + theme(aspect.ratio = .8, legend.position = c(0.09,0.15), 
                     legend.title = element_blank(),
                     legend.text = element_text(size = 11),
                     axis.title=element_text(size=15),
                     strip.text.x = element_text(size = 11)) + 
  geom_text_repel(size=3,max.overlaps = Inf) + 
  labs(x='Mean taste similarity', y='Dispersion in taste similarity', 
       fill='experts/amateurs', size='potential influence') + 
  guides(size = FALSE) +
  ylim(0.0, 0.5) + xlim(-0.1,0.7)

ggsave(file.path('results', 'visualization',
                 'fig_1.png'),scale=.75, dpi=300)
knitr::plot_crop(file.path('results','visualization', 'fig_1.png'))
fig_1





#### Figures for main paper : figure 2a ####
both_error = read.table(file.path('results','simulations',
                                  'both','performance.csv'),header=FALSE,sep=' ')
experts_error = read.table(file.path('results','simulations',
                                     'experts','performance.csv'),header=FALSE,sep=' ')
amateurs_error = read.table(file.path('results','simulations',
                                      'amateurs','performance.csv'),header=FALSE,sep=' ')
to_select = seq(5, dim(experts_error)[2], length(rho)) # select only those rhos == 1
mean_acc_e_e = colMeans(experts_error[1:NUM_EXPERTS,to_select],na.rm=T) # experts recommending experts
mean_acc_e_a = colMeans(experts_error[NUM_EXPERTS+1:NUM_TOTAL,to_select],na.rm=T) # ex reco amateurs
# delete e_e, e_a for k values > (NUM_EXPERTS -1)
mean_acc_e_e[9:17] = NA
mean_acc_e_a[9:17] = NA
mean_acc_a_e = colMeans(amateurs_error[1:NUM_EXPERTS,to_select],na.rm=T)
mean_acc_a_a = colMeans(amateurs_error[NUM_EXPERTS+1:NUM_TOTAL,to_select],na.rm=T)
mean_acc_b_e = colMeans(both_error[1:NUM_EXPERTS,to_select],na.rm=T)
mean_acc_b_a = colMeans(both_error[NUM_EXPERTS+1:NUM_TOTAL,to_select],na.rm=T)
mean_acc_all = as.data.frame(rbind(mean_acc_e_e, mean_acc_e_a, 
                                   mean_acc_a_e, mean_acc_a_a, 
                                   mean_acc_b_e, mean_acc_b_a))
colnames(mean_acc_all) = as.character(k)
mean_acc_all$reco_group = c('Experts only', 'Experts only',
                            'Amateurs only', 'Amateurs only',
                            'Both','Both')
mean_acc_all$reco_group <- factor(mean_acc_all$reco_group, levels = c("Amateurs only", "Experts only", "Both"))
mean_acc_all$stats_group = c('Expert', 'Amateur', 
                             'Expert', 'Amateur', 
                             'Expert', 'Amateur')
keycol = 'k_value'
valuecol = 'Accuracy'
longform_mean_acc_all = gather(mean_acc_all, 'k_value', 'Accuracy', as.character(k))
longform_mean_acc_all$k_value = factor(longform_mean_acc_all$k_value,
                                       levels=as.character(k))
longform_mean_acc_all$k_value = strtoi(longform_mean_acc_all$k_value)

grouped_accuracies = ggplot(data=longform_mean_acc_all, aes(x=as.numeric(k_value), y=as.numeric(Accuracy))) + 
  geom_line(aes(linetype=reco_group, color=stats_group), size=1.5, alpha=.8) + 
  scale_linetype_manual(values=c("dotdash","solid", "dashed")) + 
  scale_color_viridis(option='A',begin=COLOR_START,
                      end=.2, discrete=TRUE) + 
  theme_bw() + labs(x='k nearest neighbours',y='Accuracy') + 
  scale_x_continuous(breaks = c(1,2,3,5,7,9,11,13,15,17,19),
                     limits = c(1,19),expand = c(0.02, 0.02)) + 
  theme(aspect.ratio =1, 
        legend.position = c(0.725,0.125),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.box="horizontal") +
  scale_y_continuous(limits=c(.5,.85)) + 
  guides(color=guide_legend(title="User")) + 
  guides(linetype=guide_legend(title="Recommender")) + 
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=22))


grouped_accuracies


#### Figures for main paper : figure 2b ####
acc_figure_df <- data.frame()
stats_df$k_5_both = both_error[,26]
stats_df$k_5_experts = experts_error[,26]
stats_df$k_5_amateurs = amateurs_error[,26]
#stats_df$weighted_average = weighted_avg_error

temp_df = as.data.frame(stats_df$'k_5_experts')
temp_df$ex_am = c(rep('Expert',NUM_EXPERTS), rep('Amateur', NUM_AMATEURS))
temp_df$strategy = rep('Experts only', NUM_TOTAL)
temp_df$ids = 1:NUM_TOTAL
colnames(temp_df) = c('accuracy', 'ex_am', 'strategy', 'id')
acc_figure_df = rbind(acc_figure_df, temp_df)

temp_df = as.data.frame(stats_df$'k_5_amateurs')
temp_df$ex_am = c(rep('Expert',NUM_EXPERTS), rep('Amateur', NUM_AMATEURS))
temp_df$strategy = rep('Amateurs only', NUM_TOTAL)
temp_df$ids = 1:NUM_TOTAL
colnames(temp_df) = c('accuracy', 'ex_am', 'strategy', 'id')
acc_figure_df = rbind(acc_figure_df, temp_df)

temp_df = as.data.frame(stats_df$'k_5_both')
temp_df$ex_am = c(rep('Expert',NUM_EXPERTS), rep('Amateur', NUM_AMATEURS))
temp_df$strategy = rep('Both', NUM_TOTAL)
temp_df$ids = 1:NUM_TOTAL
colnames(temp_df) = c('accuracy', 'ex_am', 'strategy', 'id')
acc_figure_df = rbind(acc_figure_df, temp_df)

acc_figure_df$strategy = as.factor(acc_figure_df$strategy)
acc_figure_df$strategy <- factor(acc_figure_df$strategy,levels = c('Amateurs only','Experts only','Both'))

acc_fig = ggplot(data=acc_figure_df, aes(x=strategy, y=accuracy)) + 
  geom_point(aes(fill=ex_am, group=as.factor(id)), position = position_dodge(0.2), shape=21,
             size=2.5) +
  geom_line(aes(group=as.factor(id)), colour="gray",alpha=0.2,position = position_dodge(0.2)) +
  scale_fill_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') + theme_bw() + 
  labs(x='Source of recommendations',y='Accuracy') + 
  guides(fill=guide_legend(title="")) + 
  theme(aspect.ratio = 1,legend.position = c(0.2,0.9),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14)) +
  scale_y_continuous(position = "right", limits=c(.5,.85)) + 
  theme(legend.key.size=unit(.25,'cm')) + 
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=22))

g1 = ggplotGrob(grouped_accuracies)
g2 = ggplotGrob(acc_fig)
g = grid.arrange(g1,g2,nrow=1)

ggsave(file.path('results', 'visualization',
                 'fig_2.png'), g, scale=1.35)

#### Figures for main paper : figure 3b (top right) ####
expert_labels = c('WA','NM','JR',
                  'TA','B&D','JS',
                  'JL','De','RVF',
                  'JA','LeP','PW','RG','CK')
all_labels = c(expert_labels, rep('', NUM_AMATEURS))
stats_df$node_labels = all_labels

actual_influence = read.table(file.path('results','simulations','both',
                                        'adjacencies','26.csv'),
                              header=FALSE,sep=' ', row.names=NULL)
actual_influence = reshape(actual_influence, idvar = "V1", timevar = "V2", direction = "wide")
actual_influence = actual_influence[,2:dim(actual_influence)[2]]
colnames(actual_influence) = rownames(actual_influence)
stats_df$real_influence = colSums(actual_influence)
stats_df$e_a = c(rep('Experts', NUM_EXPERTS), rep('Amateurs',NUM_AMATEURS))
COLOR_START = 0.75
COLOR_END = 0.1
fig_2 = ggplot(data=stats_df, aes(y=as.numeric(potential_influence), 
                                  x=as.numeric(ratings))) + 
  geom_point(aes(fill=as.factor(e_a),size=as.numeric(ratings)),
             shape=21, alpha=0.8) + geom_text_repel(aes(label=node_labels),box.padding = 0.30,
                                                    size=7, 
                                                    max.overlaps = getOption("ggrepel.max.overlaps", default = 15)) + 
  scale_fill_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') +
  theme_bw() + theme(aspect.ratio = .8, legend.position = c(0.85,0.9), legend.title = element_blank(), 
                     legend.text = element_text(size = 10)) + 
  labs(x='Number of reviews', y='Influence potential', 
       fill='experts/amateurs', size='Actual influence') +
  guides(size = FALSE) + 
  theme(legend.text=element_text(size=16)) + 
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))

fig_2
ggsave(file.path('results','visualization', 'fig_2b.png'))



#### Supplementary figures : figure 11-12 ####
homophily_df = read.table(file.path('results',
                                    'influence.csv'),header=TRUE,sep=',')
homophily_df = subset(homophily_df, select = -c(X) )

top_panel = homophily_df
top_panel$group = as.character(homophily_df$group)
top_panel = top_panel %>% mutate(group = replace(group, group == 'experts', 'Critcs'))
top_panel = top_panel %>% mutate(group = replace(group, group == 'amateurs', 'Amateurs'))
counts = rep(c(NUM_EXPERTS, NUM_AMATEURS), nrow(top_panel)/2)

fig_11_12 = ggplot(top_panel, aes(x=kval, y=proportion_per_capita)) +  
  geom_hline(aes(yintercept=baseline_1),linetype="dashed") +
  geom_line(aes(group=rhoval,color=as.factor(rhoval))) + 
  scale_colour_viridis_d(option='F',name='rho',begin=0.0, end=0.75) + 
  ylim(0,0.07) + labs(x='Number of neighbours (k)', y='Influence') + 
  facet_grid(cols = vars(group),
             rows = vars(method)) +  theme_bw() + 
  theme(aspect.ratio = .75, legend.position=c(.075,.85)) 

addSmallLegend <- function(myPlot, pointSize = 1.25, textSize = 7, spaceLegend = 0.3) {
  myPlot +
    guides(shape = guide_legend(override.aes = list(size = pointSize)),
           color = guide_legend(override.aes = list(size = pointSize), reverse=TRUE)) +
    theme(legend.title = element_text(size = textSize), 
          legend.text  = element_text(size = textSize),
          legend.key.size = unit(spaceLegend, "lines"))
}

fig_11_12 = addSmallLegend(fig_11_12)
ggsave(file.path('results','visualization', 'fig_11_12.png'), scale=0.7)
knitr::plot_crop(file.path('results','visualization', 'fig_11_12.png'))

#### Supplementary figures : figure 10 ####
all_experts = data.frame()
expert_labels = c('WA','NM','JR',
                  'TA','B&D','JS',
                  'JL','De','RVF',
                  'JA','LeP','PW','RG','CK')
expert_labels = c('Wine Advocate','Neal Martin','Jancis Robinson',
                  'Tim Atkin','Bettane & Desseauve','James Suckling',
                  'Jeff Leve','Decanter','La Revue du Vin de France',
                  'Jane Anson','Le Point','Perswijn','Rene Gabriel','Chris Kissack')

for (r in 1:NUM_EXPERTS){ 
  rater_data = as.data.frame(correlations[r])
  rater_data$ex_am = c(rep('Expert',NUM_EXPERTS), rep('Amateur',NUM_AMATEURS))
  colnames(rater_data) = c('corrs', 'ex_am')
  rater_data$rater_name = expert_labels[r]
  all_experts = rbind(all_experts,rater_data)
}

# add an empty facet just to even things out
dummy_data = as.data.frame(correlations[1])
dummy_data$X0 = NaN
dummy_data$ex_am = NaN
colnames(dummy_data) = c('corrs', 'ex_am')
dummy_data$rater_name = ''
all_experts = rbind(all_experts,dummy_data)

all_experts$rater_name = factor(all_experts$rater_name, 
                                levels=c(expert_labels[rev(order(both_error[1:14]))], ''))


expert_hist = ggplot(all_experts, aes(x=corrs)) + 
  geom_dotplot(aes(fill=ex_am), color='gray',alpha=0.7,
               method='histodot',stackgroups=TRUE, dotsize=.7) + 
  scale_fill_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') + theme_bw() + 
  labs(x='Correlation with others', y='Number of people', fill='experts/amateurs') + 
  theme(axis.text.y = element_blank(),axis.ticks = element_blank(),aspect.ratio=.55) +
  facet_wrap(rater_name ~ ., ncol=3) + theme(legend.position = c(0.85,0.075))
expert_hist

ggsave(file.path('results','visualization', 'fig_10.png'))
knitr::plot_crop(file.path('results','visualization', 'fig_10.png'))


#### Supplementary figures : spotting cool amateurs - 1 ####
all_amateurs = data.frame()
for (r in (NUM_EXPERTS:(NUM_TOTAL-1))){ 
  rater_data = as.data.frame(correlations[r+1])
  rater_data$ex_am = c(rep('Expert',NUM_EXPERTS), rep('Amateur',NUM_AMATEURS))
  colnames(rater_data) = c('corrs', 'ex_am')
  rater_data$rater_name = r
  rater_data$mean_corr = mean(rater_data$corrs, na.rm=T)
  all_amateurs = rbind(all_amateurs,rater_data)
}

amateur_stats = stats_df[stats_df$X > (NUM_EXPERTS-1),]
sorted_amateurs = amateur_stats[order(amateur_stats$mean_corr, decreasing=TRUE), 'X']
all_amateurs$rater_name = factor(all_amateurs$rater_name, levels=sorted_amateurs)
amateur_hist = ggplot(all_amateurs, aes(x=corrs)) + 
  geom_dotplot(aes(fill=ex_am), color='gray',alpha=0.9,
               method='histodot',stackgroups=TRUE, dotsize=.7) + 
  geom_vline(aes(xintercept=mean_corr)) + 
  scale_fill_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') + theme_bw() + 
  labs(x='Correlation with others', y='Count', fill='experts/amateurs') + 
  theme(axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        aspect.ratio=.55) +
  facet_wrap(rater_name ~ ., ncol=10) +
  labs(title='Ordered by mean correlation')#crowd influence (k=133)')
amateur_hist
ggsave(file.path('results', 'visualization', 'viz_dump',
                 'mean_corr.png'),scale=2)
  
#### Supplementary figures : spotting cool amateurs-2####

homophily_df = read.table(file.path('results','homophily','both',
                                    'individual.csv'),header=TRUE,sep=',')
homophily_df = subset(homophily_df, select = -c(X) )
amateur_homophily = homophily_df[homophily_df$rater > 13,]
amateur_stats = stats_df[stats_df$X > (NUM_EXPERTS-1),]
amateur_homophily$mean_corr = rep(amateur_stats$mean_corr, 119)
amateur_homophily$disp_corr = rep(amateur_stats$disp_corr, 119)
amateur_homophily$influence = rep(amateur_stats$influence, 119)
amateur_homophily$potential_influence = rep(amateur_stats$potential_influence, 119)


sorted_amateurs = amateur_stats[order(amateur_stats$disp_corr, decreasing=TRUE), 'X']
amateur_homophily$rater = factor(amateur_homophily$rater, levels=sorted_amateurs)

amateur_line = ggplot(amateur_homophily, aes(x=as.numeric(kval),y=as.numeric(Hi))) + 
  geom_line(aes(group=rhoval,color=as.factor(rhoval))) + 
  geom_hline(aes(yintercept=baseline_2),linetype="dashed") + 
  scale_color_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') + theme_bw() + 
  labs(x='k nearest neighbour', y='Homophily index') + 
  #theme(axis.text.y = element_blank(),
  #      axis.ticks = element_blank(),
  #      aspect.ratio=.55) +
  facet_wrap(rater ~ ., ncol=10) +
  labs(title='Ordered by dispersion in correlation')#crowd influence (k=133)')
amateur_line
ggsave(file.path('results', 'visualization', 'viz_dump',
                 'homophily_disp_corr.png'),scale=2)


#### Supplementary figures : spotting cool amateurs-3####
stats_df$influence = rank(-stats_df$influence)
stats_df$bang_for_buck = rank(-stats_df$bang_for_buck)
stats_df = gather(stats_df, 'key', 'value', c('influence','bang_for_buck'))
fig = ggplot(data=stats_df, 
               aes(x=as.numeric(mean_corr), y=as.numeric(disp_corr))) + 
  geom_point(aes(fill=value),shape=21, alpha=0.7, size=4) + 
  facet_grid(cols = vars(key)) +  
  facet_grid(~factor(key, levels=c('influence',
                                   'bang_for_buck'))) + 
  scale_fill_viridis(begin=COLOR_START,end=COLOR_END, option='C') +
  theme_bw() + theme(aspect.ratio = .8, legend.position = c(0.05,0.15)) + 
  labs(x='Mean taste similarity', y='Dispersion in taste similarity', fill='Rank') + 
  ylim(0.0, 0.5) + xlim(-0.1,0.7)

fig

ggsave(file.path('results', 'visualization',
                 'fig_bang_for_buck.png'))












#### Supplementary figures : figure 5 ####
sparse_separate = read.table(file.path('results','simulations','supplementary',
                                       'sparse_separate.csv'),
                      header=FALSE,sep=' ')
sparse_together = read.table(file.path('results','simulations','supplementary',
                                       'sparse_together.csv'),
                             header=FALSE,sep=' ')

ex_mc = rowMeans(as.matrix(sparse_separate[1:NUM_EXPERTS,1:NUM_EXPERTS]),na.rm=T)
am_mc = rowMeans(as.matrix(sparse_separate[NUM_EXPERTS+1:NUM_AMATEURS,NUM_EXPERTS+1:NUM_AMATEURS]),na.rm=T)
ex_dc = rowSds(as.matrix(sparse_separate[1:NUM_EXPERTS,1:NUM_EXPERTS]),na.rm=T)
am_dc = rowSds(as.matrix(sparse_separate[NUM_EXPERTS+1:NUM_AMATEURS,NUM_EXPERTS+1:NUM_AMATEURS]),na.rm=T)

fig_5 = data.frame(as.numeric(c(ex_mc, am_mc)), as.numeric(c(ex_dc, am_dc)))
colnames(fig_5) = c('mean_corr', 'disp_corr')
both_mc = data.frame(rowMeans(as.matrix(sparse_together),na.rm=T),
                        rowSds(as.matrix(sparse_together),na.rm=T))
colnames(both_mc) = c('mean_corr', 'disp_corr')
fig_5 = rbind(fig_5, both_mc)
fig_5 = cbind(fig_5, c( rep('Source of correlations : Group members', NUM_TOTAL), 
                   rep('Source of correlations : Everyone', NUM_TOTAL) ))
colnames(fig_5)[3] = 'sep_tog'
fig_5$e_a = rep(c(rep('Experts', NUM_EXPERTS), rep('Amateurs',NUM_AMATEURS)), 2)
expert_labels = c('WA','NM','JR',
                  'TA','B&D','JS',
                  'JL','De','RVF',
                  'JA','LeP','PW','RG','CK')
all_labels = c(expert_labels, rep('', NUM_AMATEURS))
fig_5$node_labels = rep(all_labels,2)
fig_5$potential_influence = c(as.numeric(stats_df$no_influence), 
                             as.numeric(stats_df$potential_influence))

fig_5 = ggplot(data=fig_5, aes(x=as.numeric(mean_corr), y=as.numeric(disp_corr), size=potential_influence, 
                                  label=node_labels)) + 
  geom_point(aes(fill=e_a),shape=21, alpha=0.8) + 
  facet_grid(cols = vars(sep_tog)) +  
  facet_grid(~factor(sep_tog, levels=c('Source of correlations : Group members',
                                       'Source of correlations : Everyone'))) + 
  scale_fill_viridis_d(begin=COLOR_START,end=COLOR_END, option='C') +
  theme_bw() + theme(aspect.ratio = .8, legend.position = c(0.09,0.15), 
                     legend.title = element_blank(),
                     legend.text = element_text(size = 11),
                     axis.title=element_text(size=15),
                     strip.text.x = element_text(size = 11)) + 
  geom_text_repel(size=3,max.overlaps = Inf) + 
  labs(x='Mean taste similarity', y='Dispersion in taste similarity', 
       fill='experts/amateurs', size='potential influence') + 
  guides(size = FALSE) + 
  ylim(0.0, 0.5) + xlim(-0.1,0.7)

ggsave(file.path('results', 'visualization',
                 'fig_5.png'),scale=.75, dpi=300)
knitr::plot_crop(file.path('results','visualization', 'fig_5.png'))
fig_5

#### Supplementary figures : figure 6 ####
both_error = read.table(file.path('results','simulations',
                                  'both','performance.csv'),header=FALSE,sep=' ')
experts_error = read.table(file.path('results','simulations',
                                     'experts','performance.csv'),header=FALSE,sep=' ')
amateurs_error = read.table(file.path('results','simulations',
                                      'amateurs','performance.csv'),header=FALSE,sep=' ')
# pick out k = 3,5,9,13 for all errors
pick_indices = c(seq(2*length(rho)+1, (2*length(rho) + length(rho))),
                 seq(3*length(rho)+1, (3*length(rho) + length(rho))),
                 seq(5*length(rho)+1, (5*length(rho) + length(rho))),
                 seq(6*length(rho)+1, (6*length(rho) + length(rho))))
mean_acc_e_e = colMeans(experts_error[1:NUM_EXPERTS,pick_indices],na.rm=T) # experts recommending experts
mean_acc_e_a = colMeans(experts_error[NUM_EXPERTS+1:NUM_TOTAL,pick_indices],na.rm=T) # ex reco amateurs
mean_acc_a_e = colMeans(amateurs_error[1:NUM_EXPERTS,pick_indices],na.rm=T)
mean_acc_a_a = colMeans(amateurs_error[NUM_EXPERTS+1:NUM_TOTAL,pick_indices],na.rm=T)
mean_acc_b_e = colMeans(both_error[1:NUM_EXPERTS,pick_indices],na.rm=T)
mean_acc_b_a = colMeans(both_error[NUM_EXPERTS+1:NUM_TOTAL,pick_indices],na.rm=T)
mean_acc_all = as.data.frame(rbind(mean_acc_e_e, mean_acc_e_a, 
                                   mean_acc_a_e, mean_acc_a_a, 
                                   mean_acc_b_e, mean_acc_b_a))
mean_acc_all = as.matrix(mean_acc_all)
longform_mean_acc_all = data.frame()
reco_group = c('Experts only', 'Experts only',
               'Amateurs only', 'Amateurs only',
               'Both','Both')
stats_group = c('Expert', 'Amateur', 
                'Expert', 'Amateur', 
                'Expert', 'Amateur')
chosen_ks = c('k = 3','k = 5','k = 9','k = 13')
for (g_type in 1:dim(mean_acc_all)[1]){
  for (r_value in 1:dim(mean_acc_all)[2]) {
    longform_mean_acc_all = rbind(longform_mean_acc_all, 
                                  c(mean_acc_all[g_type, r_value], 
                                    reco_group[g_type], 
                                    stats_group[g_type],
                                    rho[(r_value-1)%%length(rho) + 1],
                                    chosen_ks[ceiling(r_value/length(rho))]))
  }
}

colnames(longform_mean_acc_all) = c('Accuracy', 'reco_group', 'stats_group', 'rho_value', 'k_value')
longform_mean_acc_all$rho_value = as.double(longform_mean_acc_all$rho_value)
longform_mean_acc_all$k_value = factor(longform_mean_acc_all$k_value,
                                          levels=chosen_ks)

grouped_accuracies = ggplot(data=longform_mean_acc_all, 
                            aes(x=as.numeric(rho_value), y=as.numeric(Accuracy))) + 
  facet_wrap(vars(k_value), nrow=2, ncol=2) +  
  geom_line(aes(linetype=reco_group, color=stats_group), size=.8, alpha=.8) + 
  scale_linetype_manual(values=c("dotdash","solid", "dashed")) + 
  scale_color_viridis(option='A',begin=COLOR_START,
                      end=.2, discrete=TRUE) + 
  theme_bw() + labs(x='Weighting parameter (rho)',y='Accuracy') + 
  guides(color=guide_legend(title="User")) + 
  guides(linetype=guide_legend(title="Recommender")) +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=22),
        strip.text.x = element_text(size = 11),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14))

grouped_accuracies
ggsave(file.path('results','visualization', 'fig_6.png'))
knitr::plot_crop(file.path('results','visualization', 'fig_6.png'))




#### scratch plot ####
baseline_error = read.table(file.path('results','simulations',
                                  'both','baseline.csv'),header=FALSE,sep=' ')
temp = experts_error
# select k = 1, 2, 3 ... 133 for rho = 1
#to_select = seq(5, dim(experts_error)[2], length(rho)) # select only those rhos == 1
#temp = temp[15:134,]#temp[15:134,]
temp = colMeans(temp)
#temp = temp[10,]
#temp = temp[to_select]
#temp$X = 1:134
temp = gather(as.data.frame(temp), key='k',value='v')
#temp$k = k
temp$k = rep(k, each=length(rho))
temp$rho = rep(rho, length(k))
#temp = temp[temp$rho==1,]
temp_plot = ggplot(temp, aes(x=as.numeric(rho), y=as.numeric(v))) + #geom_line(aes(color=as.factor(rho))) + 
  geom_line() +
  facet_wrap(vars(k),nrow=4, ncol=5) + 
  theme(legend.position = 'None')
temp_plot
ggsave(file.path('results','visualization', 'fig_scratch.png'))

#### scratch plot 2 ####
both_error = read.table(file.path('results','simulations',
                                  'both','performance.csv'),header=FALSE,sep=' ')
baseline_error = read.table(file.path('results','simulations',
                                      'both','baseline.csv'),header=FALSE,sep=' ')
both_error = as.matrix(both_error)
winning_strategy = data.frame()
for (u in 1:NUM_TOTAL){
  winning_strategy = rbind(winning_strategy, 
                           c(u, 
                             max(both_error[u,]),
                             k[ceiling(which.max(both_error[u,])/length(rho))],
                           rho[which.max(both_error[u,])%%length(rho) + 1],
                           baseline_error[u,]))
}
colnames(winning_strategy) = c('Rater','Accuracy','k','rho', 'baseline')

strategy_hist = ggplot(winning_strategy, aes(x=as.factor(k), 
                                             y=as.factor(rho),
                                             fill=Accuracy)) + 
  geom_tile()
strategy_hist

both_error = colMeans(both_error)
both_error = gather(as.data.frame(both_error), key='k', value='v')
both_error$k = rep(k, each=length(rho))
both_error$rho = rep(rho, length(k))
both_error = both_error[both_error$k < 23,]
k_levels = as.character(k)
r_levels = as.character(rho)
heatmap_1 = ggplot(both_error, aes(x=factor(k, levels=k_levels), 
                                   y=factor(rho, levels=r_levels), fill=v)) + 
  geom_tile() +
  scale_fill_viridis()
heatmap_1

