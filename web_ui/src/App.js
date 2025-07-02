import React, { useState, useEffect } from 'react';
import {
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Chip,
  Alert,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  InputAdornment,
  Fade,
  Grow
} from '@mui/material';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  Link as LinkIcon,
  Assessment as AssessmentIcon,
  AutoAwesome as AutoAwesomeIcon,
  Psychology as PsychologyIcon,
  School as SchoolIcon,
  Mic as MicIcon,
  MicOff as MicOffIcon
} from '@mui/icons-material';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { ThemeProvider, createTheme, alpha } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6366f1',
      light: '#818cf8',
      dark: '#4f46e5',
    },
    secondary: {
      main: '#8b5cf6',
      light: '#a78bfa',
      dark: '#7c3aed',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
    },
  },
  typography: {
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600,
          padding: '10px 24px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(99, 102, 241, 0.15)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.05)',
          },
          transition: 'box-shadow 0.3s ease-in-out',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          },
          transition: 'all 0.3s ease-in-out',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            backgroundColor: '#ffffff',
            '&:hover': {
              backgroundColor: '#f8fafc',
            },
            '&.Mui-focused': {
              backgroundColor: '#ffffff',
              '& fieldset': {
                borderColor: '#6366f1',
                borderWidth: 2,
              },
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '0.95rem',
        },
      },
    },
  },
});

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [tabValue, setTabValue] = useState(0);
  const [config, setConfig] = useState(null);
  const [citationStyle, setCitationStyle] = useState('APA');
  const [citations, setCitations] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    fetchConfig();
    fetchSessions();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await axios.get('/api/config');
      setConfig(response.data);
    } catch (error) {
      toast.error('Failed to load configuration');
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await axios.get('/api/sessions');
      setSessions(response.data.sessions);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    }
  };

  const handleResearch = async () => {
    if (!query.trim()) {
      toast.warning('Please enter a research query');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('/api/research', {
        query: query,
        use_enhanced: true,
        max_results: 15
      });

      setResults(response.data);
      setTabValue(0);
      toast.success('Research completed successfully!');
      fetchSessions();
    } catch (error) {
      toast.error('Research failed: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format) => {
    if (!results?.session_id) {
      toast.warning('No results to export');
      return;
    }

    try {
      const response = await axios.get(`/api/export/${results.session_id}?format=${format}`);
      const blob = new Blob([response.data.data], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research_results.${format}`;
      a.click();
      window.URL.revokeObjectURL(url);
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Export failed: ' + error.message);
    }
  };

  const fetchCitations = async () => {
    try {
      const response = await axios.get(`/api/citations?style=${citationStyle}&format=text`);
      setCitations(response.data.citations);
    } catch (error) {
      toast.error('Failed to fetch citations');
    }
  };

  const renderResults = () => {
    if (!results) return null;

    return (
      <Box>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Grow in timeout={600}>
              <Card 
                sx={{ 
                  borderRadius: 3,
                  border: '1px solid rgba(99, 102, 241, 0.08)',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                  },
                }}
              >
                <CardContent sx={{ p: 4 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                    <Box
                      sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <SchoolIcon sx={{ color: 'white', fontSize: 24 }} />
                    </Box>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      Research Summary
                    </Typography>
                  </Box>
                  
                  <Box sx={{ 
                    backgroundColor: alpha(theme.palette.primary.main, 0.02),
                    borderRadius: 2,
                    p: 3,
                    mb: 3,
                  }}>
                    <ReactMarkdown>{results.summary}</ReactMarkdown>
                  </Box>
                  
                  {results.confidence_score && (
                    <Box sx={{ mt: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          Confidence Score
                        </Typography>
                        <Typography variant="body2" fontWeight={600} color="primary">
                          {(results.confidence_score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={results.confidence_score * 100} 
                        sx={{ 
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: alpha(theme.palette.primary.main, 0.1),
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 4,
                            background: 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)',
                          },
                        }}
                      />
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grow>
          </Grid>

          {results.key_findings && results.key_findings.length > 0 && (
            <Grid item xs={12}>
              <Grow in timeout={800}>
                <Card 
                  sx={{ 
                    borderRadius: 3,
                    border: '1px solid rgba(99, 102, 241, 0.08)',
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                      ✨ Key Findings
                    </Typography>
                    <Grid container spacing={2}>
                      {results.key_findings.map((finding, index) => (
                        <Grid item xs={12} sm={6} key={index}>
                          <Box
                            sx={{
                              p: 2,
                              borderRadius: 2,
                              backgroundColor: alpha(theme.palette.primary.main, 0.04),
                              border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                              '&:hover': {
                                backgroundColor: alpha(theme.palette.primary.main, 0.08),
                                transform: 'translateX(4px)',
                              },
                              transition: 'all 0.2s ease',
                            }}
                          >
                            <Typography variant="body2">
                              {finding}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grow>
            </Grid>
          )}

          <Grid item xs={12}>
            <Grow in timeout={1000}>
              <Card 
                sx={{ 
                  borderRadius: 3,
                  border: '1px solid rgba(99, 102, 241, 0.08)',
                }}
              >
                <CardContent sx={{ p: 4 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                    <LinkIcon sx={{ color: 'primary.main' }} />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Sources ({results.sources?.length || 0})
                    </Typography>
                  </Box>
                  <Grid container spacing={2}>
                    {results.sources?.map((source, index) => (
                      <Grid item xs={12} md={6} key={index}>
                        <Box
                          sx={{
                            p: 3,
                            borderRadius: 2,
                            backgroundColor: 'background.default',
                            border: '1px solid rgba(0, 0, 0, 0.06)',
                            height: '100%',
                            display: 'flex',
                            flexDirection: 'column',
                            '&:hover': {
                              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                              transform: 'translateY(-2px)',
                            },
                            transition: 'all 0.2s ease',
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 2 }}>
                            <Box
                              sx={{
                                minWidth: 32,
                                height: 32,
                                borderRadius: 1,
                                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <LinkIcon sx={{ fontSize: 18, color: 'primary.main' }} />
                            </Box>
                            <Box sx={{ flex: 1 }}>
                              <Typography 
                                component="a" 
                                href={source.url} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                sx={{ 
                                  textDecoration: 'none',
                                  color: 'primary.main',
                                  fontWeight: 600,
                                  '&:hover': {
                                    textDecoration: 'underline',
                                  },
                                }}
                              >
                                {source.title}
                              </Typography>
                              {source.credibility > 0.7 && (
                                <Chip 
                                  label="High Credibility" 
                                  size="small" 
                                  sx={{ 
                                    ml: 1,
                                    backgroundColor: alpha(theme.palette.success.main, 0.1),
                                    color: 'success.main',
                                  }} 
                                />
                              )}
                            </Box>
                          </Box>
                          <Typography 
                            variant="body2" 
                            color="text.secondary"
                            sx={{ 
                              flex: 1,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              display: '-webkit-box',
                              WebkitLineClamp: 3,
                              WebkitBoxOrient: 'vertical',
                            }}
                          >
                            {source.snippet || source.content}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grow>
          </Grid>

          {results.gaps_identified && results.gaps_identified.length > 0 && (
            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="subtitle1" gutterBottom>
                  Research Gaps Identified:
                </Typography>
                <ul>
                  {results.gaps_identified.map((gap, index) => (
                    <li key={index}>{gap}</li>
                  ))}
                </ul>
              </Alert>
            </Grid>
          )}
        </Grid>
      </Box>
    );
  };

  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);

  useEffect(() => {
    // Check if speech recognition is supported
    setSpeechSupported('webkitSpeechRecognition' in window || 'SpeechRecognition' in window);
  }, []);

  const handleVoiceInput = () => {
    if (!speechSupported) {
      toast.error('Speech recognition is not supported in your browser');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
      toast.info('Listening... Speak your research question');
    };

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map(result => result[0])
        .map(result => result.transcript)
        .join('');
      
      setQuery(transcript);
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      toast.error('Speech recognition error: ' + event.error);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', backgroundColor: 'background.default' }}>
        <AppBar 
          position="static" 
          elevation={0}
          sx={{ 
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid rgba(0, 0, 0, 0.05)',
          }}
        >
          <Toolbar>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexGrow: 1 }}>
              <PsychologyIcon sx={{ color: 'primary.main', fontSize: 32 }} />
              <Typography 
                variant="h6" 
                component="div" 
                sx={{ 
                  color: 'text.primary',
                  fontWeight: 700,
                  background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Research Agent 2
              </Typography>
            </Box>
            <IconButton 
              onClick={() => setSettingsOpen(true)}
              sx={{ 
                color: 'text.secondary',
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.08),
                },
              }}
            >
              <SettingsIcon />
            </IconButton>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 6, mb: 4 }}>
          <Fade in timeout={800}>
            <Paper 
              elevation={0} 
              sx={{ 
                p: 5, 
                mb: 4,
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%)',
                border: '1px solid rgba(99, 102, 241, 0.1)',
              }}
            >
              <Box sx={{ textAlign: 'center', mb: 4 }}>
                <AutoAwesomeIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography 
                  variant="h4" 
                  gutterBottom
                  sx={{
                    background: 'linear-gradient(135deg, #1e293b 0%, #475569 100%)',
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  Intelligent Research Assistant
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Ask anything and get comprehensive, well-researched answers
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="What would you like to research today?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleResearch()}
                  disabled={loading || isListening}
                  multiline
                  rows={2}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: 'white',
                    },
                  }}
                  InputProps={{
                    endAdornment: speechSupported && (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={handleVoiceInput}
                          edge="end"
                          sx={{
                            color: isListening ? 'primary.main' : 'text.secondary',
                            animation: isListening ? 'pulse 1.5s infinite' : 'none',
                            '@keyframes pulse': {
                              '0%': { opacity: 1 },
                              '50%': { opacity: 0.5 },
                              '100%': { opacity: 1 },
                            },
                          }}
                        >
                          {isListening ? <MicIcon /> : <MicOffIcon />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
                <Button
                  variant="contained"
                  onClick={handleResearch}
                  disabled={loading || isListening}
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                  sx={{ 
                    minWidth: 150,
                    height: 56,
                    fontSize: '1rem',
                  }}
                >
                  {loading ? 'Researching...' : 'Research'}
                </Button>
              </Box>

              {config && (
                <Box sx={{ mt: 3, display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
                  {Object.entries(config.apis_configured).map(([api, configured]) => (
                    <Chip
                      key={api}
                      label={api}
                      color={configured ? 'primary' : 'default'}
                      size="small"
                      sx={{
                        backgroundColor: configured 
                          ? alpha(theme.palette.primary.main, 0.1)
                          : alpha(theme.palette.text.secondary, 0.1),
                        color: configured 
                          ? 'primary.main'
                          : 'text.secondary',
                      }}
                    />
                  ))}
                </Box>
              )}
            </Paper>
          </Fade>

          <Paper 
            elevation={0}
            sx={{ 
              width: '100%',
              borderRadius: 3,
              border: '1px solid rgba(0, 0, 0, 0.08)',
              overflow: 'hidden',
            }}
          >
            <Tabs 
              value={tabValue} 
              onChange={(e, v) => setTabValue(v)}
              sx={{
                borderBottom: '1px solid rgba(0, 0, 0, 0.08)',
                '& .MuiTab-root': {
                  minHeight: 64,
                },
                '& .Mui-selected': {
                  color: 'primary.main',
                },
                '& .MuiTabs-indicator': {
                  height: 3,
                  borderRadius: '3px 3px 0 0',
                  background: 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)',
                },
              }}
            >
              <Tab 
                label="Results" 
                icon={<AssessmentIcon />} 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                label="Citations" 
                icon={<LinkIcon />} 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                label="History" 
                icon={<HistoryIcon />} 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              {renderResults()}
              {results && (
                <Fade in timeout={1200}>
                  <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
                    <Button
                      variant="outlined"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleExport('json')}
                      sx={{
                        borderColor: alpha(theme.palette.primary.main, 0.3),
                        color: 'primary.main',
                        '&:hover': {
                          borderColor: 'primary.main',
                          backgroundColor: alpha(theme.palette.primary.main, 0.04),
                        },
                      }}
                    >
                      Export JSON
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleExport('markdown')}
                      sx={{
                        borderColor: alpha(theme.palette.primary.main, 0.3),
                        color: 'primary.main',
                        '&:hover': {
                          borderColor: 'primary.main',
                          backgroundColor: alpha(theme.palette.primary.main, 0.04),
                        },
                      }}
                    >
                      Export Markdown
                    </Button>
                  </Box>
                </Fade>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
                <FormControl variant="outlined" size="small">
                  <InputLabel>Citation Style</InputLabel>
                  <Select
                    value={citationStyle}
                    onChange={(e) => setCitationStyle(e.target.value)}
                    label="Citation Style"
                  >
                    <MenuItem value="APA">APA</MenuItem>
                    <MenuItem value="MLA">MLA</MenuItem>
                    <MenuItem value="CHICAGO">Chicago</MenuItem>
                    <MenuItem value="IEEE">IEEE</MenuItem>
                  </Select>
                </FormControl>
                <Button variant="contained" onClick={fetchCitations}>
                  Generate Citations
                </Button>
              </Box>
              {citations && (
                <Paper sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
                  <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                    {citations}
                  </pre>
                </Paper>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <List>
                {sessions.map((session) => (
                  <ListItem key={session.session_id}>
                    <ListItemText
                      primary={session.query}
                      secondary={`Started: ${new Date(session.started_at).toLocaleString()} - Status: ${session.status}`}
                    />
                  </ListItem>
                ))}
              </List>
            </TabPanel>
          </Paper>
        </Container>

        <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Settings</DialogTitle>
          <DialogContent>
            <Alert severity="info" sx={{ mb: 2 }}>
              Configure API keys in config_local.json file
            </Alert>
            <Typography variant="body2" paragraph>
              To enable additional search capabilities, add your API keys to the config_local.json file:
            </Typography>
            <Paper sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
              <pre style={{ fontSize: '0.875rem' }}>
{`{
  "apis": {
    "newsapi": {
      "auth_config": {
        "api_key": "your_api_key_here"
      }
    }
  }
}`}
              </pre>
            </Paper>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSettingsOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>

        <ToastContainer position="bottom-right" />
      </Box>
    </ThemeProvider>
  );
}

export default App;