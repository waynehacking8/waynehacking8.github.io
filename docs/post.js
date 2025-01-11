class PostPage {
    constructor() {
        this.postContent = document.getElementById('post-content');
        marked.setOptions({
            headerIds: false,
            mangle: false,
            headerPrefix: '',
            breaks: true,
            gfm: true
        });
        this.init();
    }

    async init() {
        try {
            const urlParams = new URLSearchParams(window.location.search);
            const postId = urlParams.get('id');
            if (!postId) {
                this.showError('找不到文章');
                return;
            }

            await this.loadPost(postId);
        } catch (error) {
            console.error('載入文章失敗:', error);
            this.showError('載入失敗，請重試');
        }
    }

    async loadPost(postId) {
        const response = await fetch('./posts.json');
        if (!response.ok) throw new Error('無法載入文章');
        
        const posts = await response.json();
        const post = posts.find(p => p.id === postId);
        
        if (!post) {
            this.showError('找不到文章');
            return;
        }

        this.displayPost(post);
        document.title = post.title; // 更新頁面標題
    }

    displayPost(post) {
        const renderer = new marked.Renderer();
        renderer.heading = (text, level) => {
            return `<h${level} style="color: var(--color-primary);">${text}</h${level}>`;
        };
        renderer.paragraph = (text) => {
            return `<p style="color: var(--color-text);">${text}</p>`;
        };
        renderer.list = (body, ordered) => {
            const type = ordered ? 'ol' : 'ul';
            return `<${type} style="color: var(--color-text);">${body}</${type}>`;
        };
        renderer.listitem = (text) => {
            return `<li style="color: var(--color-text);">${text}</li>`;
        };
        
        marked.setOptions({ renderer });

        this.postContent.innerHTML = `
            <div class="post-header">
                <h1>${post.title}</h1>
                <div class="post-meta">
                    <span class="post-date">
                        <i class="far fa-calendar"></i>
                        ${new Date(post.date).toLocaleDateString('zh-TW')}
                    </span>
                    <span class="post-category">
                        <i class="fas ${this.getCategoryIcon(post.category)}"></i>
                        ${this.formatCategory(post.category)}
                    </span>
                </div>
                <div class="post-tags">
                    ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
            <div class="post-content-full">
                ${marked.parse(post.content)}
            </div>
        `;
    }

    formatCategory(category) {
        const categoryMap = {
            'machine-learning': '機器學習',
            'cybersecurity': '資安',
            'programming': '程式開發',
            'others': '其他'
        };
        return categoryMap[category] || category;
    }

    getCategoryIcon(category) {
        const icons = {
            'machine-learning': 'fa-brain',
            'cybersecurity': 'fa-shield-alt',
            'programming': 'fa-code',
            'others': 'fa-ellipsis-h'
        };
        return icons[category] || 'fa-file-alt';
    }

    showError(message) {
        this.postContent.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                ${message}
            </div>
        `;
    }
}

// 初始化文章頁面
document.addEventListener('DOMContentLoaded', () => {
    new PostPage();
}); 