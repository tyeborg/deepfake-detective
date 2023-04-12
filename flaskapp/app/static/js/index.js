// Enable cool fade in when page refreshes.
const sr = ScrollReveal ({
    distance: '40px',
    duration: 2500,
    reset: true
});

sr.reveal('.logo', {delay:400, origin: 'left'});
sr.reveal('.homeText span', {delay:600, origin: 'top'});
sr.reveal('.homeText h2', {delay:600, origin: 'left'});
sr.reveal('.homeText h3', {delay:600, origin: 'left'});
sr.reveal('.homeText p', {delay:680, origin: 'right'});
sr.reveal('.mainBtn', {delay:750, origin: 'left'});